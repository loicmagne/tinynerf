from src.models import VanillaFeatureMLP, VanillaOpacityDecoder, VanillaColorDecoder
from src.models import KPlanesFeatureField, KPlanesExplicitOpacityDecoder, KPlanesExplicitColorDecoder
from src.models import CobafaFeatureField
from src.data import ImagesDataset, RaysDataset
from src.core import OccupancyGrid, NerfRenderer, RayMarcherAABB, RayMarcherUnbounded, ContractionAABB, ContractionMip360, RayProvider, RayMarcher, Contraction
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Tuple
from PIL import Image
from pathlib import Path
import json
import torch

def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return - 10. * torch.log10(torch.mean((x - y) ** 2))

@dataclass
class TrainMetrics:
    loss: float = 0.
    occupancy: float = 1.

@dataclass
class TestMetrics:
    loss: float = 0.
    psnr: float = 0.
    ssim: float = 0.

@dataclass
class VanillaTrainConfig:
    train_rays: RaysDataset
    test_images: ImagesDataset
    output: Path
    method: str
    batch_size: int
    n_samples: int
    eval_every: int
    eval_n : int
    scene_type: str


def train_vanilla(cfg: VanillaTrainConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Compute bunch of constant given we target 30k iter @ 4096 batch size
    bs_ratio = 4096 / cfg.batch_size

    steps = int(30000 * bs_ratio)

    occupancy_grid_updates = int(16 * bs_ratio)
    occupancy_grid_threshold = 0.01
    occupancy_res = 128
    # heuristic: a voxel must be queried 16 times empty to be considered empty
    occupancy_grid_decay = occupancy_grid_threshold ** (1 / 16)

    lr_init = 1e-2
    weight_decay = 1e-5
    tv_reg_alpha = 0.0001
    l1_reg_alpha = 1e-3

    train_loader = DataLoader(cfg.train_rays, cfg.batch_size, shuffle=True)

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
            coef_res=128,
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
        ray_marcher = RayMarcherUnbounded(cfg.n_samples, 0.1, 1e5, uniform_range=cfg.train_rays.scene_scale)
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

    ray_provider = RayProvider(
        occupancy_grid=occupancy_grid,
        contraction=contraction,
        ray_marcher=ray_marcher
    )

    renderer = NerfRenderer(
        feature_module=feature_module,
        sigma_decoder=sigma_decoder,
        rgb_decoder=rgb_decoder,
        bg_color=cfg.train_rays.bg_color
    ).to(device)

    print(f'Using {cfg.method} with {sum(p.numel() for p in renderer.parameters())} parameters.')    

    optimizer = torch.optim.Adam(renderer.parameters(), lr=lr_init, eps=1e-15, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ChainedScheduler([
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                steps // 2,
                steps * 3 // 4,
                steps * 5 // 6,
                steps * 9 // 10,
            ],
            gamma=0.33,
        ),
    ])

    scaler = torch.cuda.amp.GradScaler(2 ** 10) # type: ignore
    loss_fn = torch.nn.MSELoss()

    def eval_step(img_dataset: ImagesDataset, indices: List[int], name: str):
        renderer.eval()
        metrics_acc: List[TestMetrics] = []
        with torch.no_grad():
            for i in tqdm(indices):
                metrics = TestMetrics()
                data = img_dataset[i % len(img_dataset)]
                img = data['rgbs']
                rays_o = data['rays_o'].view(-1, 3)
                rays_d = data['rays_d'].view(-1, 3)
                rgbs = data['rgbs'].view(-1, 3)
                rendered_rgbs = []
                for k in range(0, len(rays_o), cfg.batch_size):
                    start = k
                    end = k + cfg.batch_size
                    batch_rays_o = rays_o[start:end].to(device)
                    batch_rays_d = rays_d[start:end].to(device)
                    batch_rgbs = rgbs[start:end].to(device)
                    samples, info = ray_provider(batch_rays_o, batch_rays_d, training=False)

                    batch_rendered_rgbs = renderer(samples, info)
                    metrics.loss += torch.sum((batch_rendered_rgbs - batch_rgbs)**2).item()
                    rendered_rgbs.append(batch_rendered_rgbs.cpu())
                rendered_img = torch.cat(rendered_rgbs, dim=0).view(img.shape)
                metrics.psnr = psnr(img, rendered_img).item()
                metrics.loss /= rays_o.size(0)
                metrics_acc.append(metrics)

                # Save image
                rendered_img = (255. * rendered_img).type(torch.uint8).numpy()
                Image.fromarray(rendered_img).save(cfg.output / f'{name}_{i}.png')
        return metrics_acc

    def loop() -> Tuple[List[TrainMetrics], List[TestMetrics], List[TestMetrics]]:
        train_iter = iter(train_loader)
        train_metrics: List[TrainMetrics] = []
        test_metrics: List[TestMetrics] = []
        train_step = 0
        test_step = 0
        target_sample_size = cfg.batch_size * cfg.n_samples
        with tqdm(total=steps) as pbar:
            while True:
                # Dynamically generate a batch of samples
                with torch.no_grad():
                    current_size = 0.
                    projected_size = 0.
                    tmp_count = 0
                    acc_info, acc_samples, acc_rgbs = [], [], []
                    while projected_size < target_sample_size:
                        try:
                            data = next(train_iter)
                        except StopIteration:
                            train_iter = iter(train_loader)
                            data = next(train_iter)
                        rays_o = data['rays_o'].to(device)
                        rays_d = data['rays_d'].to(device)
                        rgbs = data['rgbs'].to(device)
                        samples, info = ray_provider(rays_o, rays_d, training=True)

                        acc_info.append(info)
                        acc_samples.append(samples)
                        acc_rgbs.append(rgbs)

                        current_size += samples.size(0)
                        tmp_count += 1
                        # To know if we should run another iteration we add the average number of samples
                        # generated from each batch to the current size
                        projected_size = current_size * (1 + 1/tmp_count)
                    packed_samples = torch.cat(acc_samples, 0)
                    packed_rgbs = torch.cat(acc_rgbs, 0)
                    packing_info = torch.cat(acc_info, 0)

                renderer.train()

                if train_step % occupancy_grid_updates == 0:
                    occupancy_grid.update(lambda t: renderer.sigma_decoder(renderer.feature_module(t)))

                rendered_rgbs = renderer(packed_samples, packing_info)
                loss = loss_fn(rendered_rgbs,packed_rgbs)

                if cfg.method == 'kplanes':
                    loss += renderer.feature_module.loss_tv() * tv_reg_alpha # type: ignore
                    loss += renderer.feature_module.loss_l1() * l1_reg_alpha # type: ignore

                optimizer.zero_grad()
                scaler.scale(loss).backward() # type: ignore

                for name, param in renderer.named_parameters():
                    print(f'{name}: {param.grad}')

                optimizer.step()
                scheduler.step()

                train_loss = loss.detach().cpu().item()
                occupancy = occupancy_grid.occupancy()
                train_metrics.append(TrainMetrics(train_loss, occupancy))
                pbar.set_postfix(
                    loss=train_loss,
                    occupancy=occupancy,
                    rendered_samples=packed_samples.size(0) / target_sample_size
                )

                if train_step % cfg.eval_every == 0 and train_step > 0:
                    indices = list(range(test_step, test_step + cfg.eval_n))
                    metrics = eval_step(cfg.test_images, indices, f'test_{train_step}')
                    test_metrics.extend(metrics)
                    test_step += cfg.eval_n

                if train_step >= steps:
                    # Evaluate on full test set
                    indices = list(range(len(cfg.test_images)))
                    final_metrics = eval_step(cfg.test_images, indices, f'test_full')
                    # Save model
                    torch.save(renderer.state_dict(), cfg.output / 'model.pt')
                    return train_metrics, test_metrics, final_metrics

                train_step += 1
                pbar.update(1)

    train_metrics, test_metrics, final_metrics = loop()
    json.dump([asdict(x) for x in train_metrics], open(cfg.output / 'metrics_train.json', 'w'))
    json.dump([asdict(x) for x in test_metrics], open(cfg.output / 'metrics_test.json', 'w'))
    json.dump([asdict(x) for x in final_metrics], open(cfg.output / 'metrics_final.json', 'w'))