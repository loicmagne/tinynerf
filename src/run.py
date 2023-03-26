from src.models import VanillaFeatureMLP, VanillaOpacityDecoder, VanillaColorDecoder
from src.models import KPlanesFeatureField, KPlanesExplicitOpacityDecoder, KPlanesExplicitColorDecoder
from src.models import CobafaFeatureField
from src.data import NerfData
from src.core import OccupancyGrid, NerfRenderer, RayMarcherAABB, RayMarcherUnbounded, ContractionAABB, ContractionMip360, SamplesProvider, samples_generator, RayMarcher, Contraction
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Tuple
from PIL import Image
from pathlib import Path
import json
import torch

def infer(
    renderer: NerfRenderer,
    provider: SamplesProvider,
    dataset: NerfData,
    indices: List[int],
    folder: Path,
    name: str,
    device: torch.device,
    target_size: int,
    batch_size: int = 2048,
):
    renderer.eval()
    rendered: List[torch.Tensor] = []
    with torch.no_grad():
        for i in tqdm(indices):
            data = dataset[i]
            rays_o = data['rays_o'].view(-1, 3)
            rays_d = data['rays_d'].view(-1, 3)
            intrinsics = dataset.idx_intrinsics(i)
            img_shape = [intrinsics.h, intrinsics.w, 3]
            rendered_rgbs = []
            for (
                packed_samples,
                packing_info,
                _
            ) in samples_generator(
                rays_o, rays_d, None,
                provider,
                target_size,
                batch_size,
                False, device
            ):
                batch_rendered_rgbs = renderer(packed_samples, packing_info)
                rendered_rgbs.append(batch_rendered_rgbs.cpu())
 
            rendered_img = torch.cat(rendered_rgbs, dim=0).view(img_shape)
            rendered.append(rendered_img.cpu().clone())
            # Save image
            rendered_img = (255. * rendered_img).type(torch.uint8).numpy()
            Image.fromarray(rendered_img).save(folder / f'{name}_{i}.png')
    return rendered


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return - 10. * torch.log10(torch.mean((x - y) ** 2))

@dataclass
class EvalMetrics:
    mse_loss: float = 0.
    psnr: float = 0.
    ssim: float = 0.

def eval(
    dataset: NerfData,
    rendered_imgs: List[torch.Tensor],
    indices: List[int]
):
    assert dataset.imgs is not None
    metrics_acc: List[EvalMetrics] = []
    with torch.no_grad():
        for i, rendered_img in zip(indices, rendered_imgs):
            metrics = EvalMetrics()
            true_img = dataset[i]['rgbs']
            metrics.mse_loss = torch.nn.functional.mse_loss(true_img, rendered_img).item()
            metrics.psnr = psnr(true_img, rendered_img).item()
            metrics_acc.append(metrics)
    return metrics_acc

@dataclass
class TrainMetrics:
    loss: float = 0.
    occupancy: float = 1.

@dataclass
class TrainConfig:
    method: str
    train_set: NerfData
    eval_set: NerfData | None
    eval_every: int | None
    eval_n : int | None
    test_set: NerfData | None
    scene_type: str
    output: Path
    batch_size: int
    n_samples: int


def train(cfg: TrainConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Compute bunch of constant given we target 30k iter @ 4096 batch size
    bs_ratio = 4096 / cfg.batch_size

    total_steps = int(2048 * bs_ratio)

    occupancy_grid_updates = int(16 * bs_ratio)
    occupancy_grid_threshold = 0.01
    occupancy_res = 128
    # heuristic: a voxel must be queried 16 times empty to be considered empty
    occupancy_grid_decay = occupancy_grid_threshold ** (1 / 16)

    lr_init = 1e-2
    weight_decay = 1e-5
    tv_reg_alpha = 0.0001
    l1_reg_alpha = 0.

    feature_module: torch.nn.Module
    sigma_decoder: torch.nn.Module
    rgb_decoder: torch.nn.Module
    ray_marcher: RayMarcher
    contraction: Contraction

    if cfg.method == 'vanilla':
        feature_module = VanillaFeatureMLP(10, 256, 8)
        dim = feature_module.feature_dim
        sigma_decoder = VanillaOpacityDecoder(dim)
        rgb_decoder = VanillaColorDecoder(8, dim, 64, 3)
    elif cfg.method == 'kplanes':
        feature_module = KPlanesFeatureField(32)
        dim = feature_module.feature_dim
        sigma_decoder = VanillaOpacityDecoder(dim)
        rgb_decoder = VanillaColorDecoder(8, dim, 64, 3)
    elif cfg.method == 'cobafa':
        feature_module = CobafaFeatureField(
            basis_res=torch.linspace(32., 128, 6).int().tolist(),
            coef_res=64,
            freqs=torch.linspace(2., 8., 6).tolist(),
            channels=[8,8,8,4,4,4],
            mlp_hidden_dim=128
        )
        dim = feature_module.feature_dim
        sigma_decoder = VanillaOpacityDecoder(dim)
        rgb_decoder = VanillaColorDecoder(8, dim, 64, 3)
    else:
        raise NotImplementedError(f'Unknown method {cfg.method}.')
    
    if cfg.scene_type == 'unbounded':
        ray_marcher = RayMarcherUnbounded(cfg.n_samples, 0.1, 1e5, uniform_range=1.)
        contraction = ContractionMip360(order=float('inf'))
    elif cfg.scene_type == 'aabb':
        aabb = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]).to(device)
        ray_marcher = RayMarcherAABB(aabb, cfg.n_samples, 0.1)
        contraction = ContractionAABB(aabb)
    else:
        raise NotImplementedError(f'Unknown scene type {cfg.scene_type}.')

    occupancy_grid = OccupancyGrid(
        size=occupancy_res,
        step_size=ray_marcher.step_size,
        threshold=occupancy_grid_threshold,
        decay=occupancy_grid_decay
    ).to(device)

    samples_provider = SamplesProvider(
        occupancy_grid=occupancy_grid,
        contraction=contraction,
        ray_marcher=ray_marcher
    )

    renderer = NerfRenderer(
        feature_module=feature_module,
        sigma_decoder=sigma_decoder,
        rgb_decoder=rgb_decoder,
        bg_color=cfg.train_set.bg_color
    ).to(device)

    print(f'Using {cfg.method} with {sum(p.numel() for p in renderer.parameters())} parameters.')    

    optimizer = torch.optim.Adam(
        renderer.parameters(),
        lr=lr_init,
        eps=1e-15,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ChainedScheduler([
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                total_steps // 2,
                total_steps * 3 // 4,
                total_steps * 5 // 6,
                total_steps * 9 // 10,
            ],
            gamma=0.33,
        ),
    ])

    scaler = torch.cuda.amp.GradScaler(2 ** 10) # type: ignore
    loss_fn = torch.nn.MSELoss()

    def loop() -> Tuple[List[TrainMetrics], List[EvalMetrics] | None, List[EvalMetrics] | None]:
        acc_train_metrics: List[TrainMetrics] = []
        acc_eval_metrics: List[EvalMetrics] = []
        steps = 0
        eval_steps = 0
        target_sample_size = cfg.batch_size * cfg.n_samples
        with tqdm(total=total_steps) as pbar:
            while True:
                data = cfg.train_set[steps % len(cfg.train_set)]
                rays_o = data['rays_o'].view(-1, 3)
                rays_d = data['rays_d'].view(-1, 3)
                rgbs = data['rgbs'].view(-1, 3)
                for (
                    packed_samples,
                    packing_info,
                    packed_rgbs
                ) in samples_generator(
                    rays_o, rays_d, rgbs,
                    samples_provider,
                    target_sample_size,
                    cfg.batch_size,
                    True, device
                ):
                    renderer.train()

                    if steps % occupancy_grid_updates == 0:
                        occupancy_grid.update(lambda t: renderer.sigma_decoder(renderer.feature_module(t)))

                    rendered_rgbs = renderer(packed_samples, packing_info)
                    loss = loss_fn(rendered_rgbs,packed_rgbs)

                    if cfg.method == 'kplanes':
                        loss += renderer.feature_module.loss_tv() * tv_reg_alpha # type: ignore
                        loss += renderer.feature_module.loss_l1() * l1_reg_alpha # type: ignore

                    optimizer.zero_grad()
                    scaler.scale(loss).backward() # type: ignore
                    optimizer.step()
                    scheduler.step()

                    train_metrics = TrainMetrics()
                    train_metrics.loss = loss.detach().cpu().item()
                    train_metrics.occupancy = occupancy_grid.occupancy()
                    acc_train_metrics.append(train_metrics)
                    pbar.set_postfix(
                        loss=train_metrics.loss,
                        occupancy=train_metrics.occupancy,
                        rendered_samples=packed_samples.size(0) / target_sample_size
                    )

                    if cfg.eval_every is not None and cfg.eval_n is not None and cfg.eval_set is not None:
                        if steps % cfg.eval_every == 0 and steps > 0:
                            indices = list(range(eval_steps, eval_steps + cfg.eval_n))
                            eval_rendered = infer(
                                renderer=renderer,
                                provider=samples_provider,
                                dataset=cfg.eval_set,
                                indices=indices,
                                folder=cfg.output,
                                name=f'test_{steps}',
                                device=device,
                                target_size=target_sample_size,
                                batch_size=cfg.batch_size
                            )
                            metrics = eval(cfg.eval_set, eval_rendered, indices)
                            acc_eval_metrics.extend(metrics)
                            eval_steps += cfg.eval_n

                    if steps >= total_steps:
                        test_metrics = None
                        if cfg.test_set is not None:
                            # Evaluate on full test set
                            indices = list(range(len(cfg.test_set)))
                            test_rendered = infer(
                                renderer=renderer,
                                provider=samples_provider,
                                dataset=cfg.test_set,
                                indices=indices,
                                folder=cfg.output,
                                name=f'test_full',
                                device=device,
                                target_size=target_sample_size,
                                batch_size=cfg.batch_size
                            )
                            if cfg.test_set.imgs:
                                test_metrics = eval(cfg.test_set, test_rendered, indices)
                        # Save model
                        torch.save(renderer.state_dict(), cfg.output / 'model.pt')
                        return acc_train_metrics, acc_eval_metrics, test_metrics

                    steps += 1
                    pbar.update(1)

    train_metrics, eval_metrics, test_metrics = loop()
    json.dump([asdict(x) for x in train_metrics], open(cfg.output / 'metrics_train.json', 'w'))
    if eval_metrics:
        json.dump([asdict(x) for x in eval_metrics], open(cfg.output / 'metrics_eval.json', 'w'))
    if test_metrics:
        json.dump([asdict(x) for x in test_metrics], open(cfg.output / 'metrics_test.json', 'w'))

