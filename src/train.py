from src.models import VanillaFeatureMLP, VanillaOpacityDecoder, VanillaColorDecoder
from src.models import KPlanesFeatureField, KPlanesExplicitOpacityDecoder, KPlanesExplicitColorDecoder
from src.models import KPlanesHybridOpacityDecoder, KPlanesHybridColorDecoder
from src.data import ImagesDataset, RaysDataset
from src.core import OccupancyGrid, NerfRenderer, RayMarcherAABB, RayMarcherUnbounded, ContractionAABB, ContractionMip360, RayMarcher, Contraction
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
    occupancy_grid_decay  = occupancy_grid_threshold ** (1 / 16)

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
        feature_module = VanillaFeatureMLP(10, [256 for _ in range(8)])
        dim = feature_module.feature_dim
        sigma_decoder = VanillaOpacityDecoder(dim)
        rgb_decoder = VanillaColorDecoder(4, dim, [128])
    elif cfg.method == 'kplanes':
        feature_module = KPlanesFeatureField(32)
        dim = feature_module.feature_dim
        sigma_decoder = KPlanesHybridOpacityDecoder(dim)
        rgb_decoder = KPlanesHybridColorDecoder(dim)
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
    )

    renderer = NerfRenderer(
        occupancy_grid=occupancy_grid,
        feature_module=feature_module,
        sigma_decoder=sigma_decoder,
        rgb_decoder=rgb_decoder,
        contraction=contraction,
        ray_marcher=ray_marcher,
        bg_color=cfg.train_rays.bg_color
    ).to(device)

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
            metrics = TestMetrics()
            for i in tqdm(indices):
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

                    batch_rendered_rgbs, _ = renderer(batch_rays_o, batch_rays_d)
                    metrics.loss += torch.sum((batch_rendered_rgbs - batch_rgbs)**2).item()
                    rendered_rgbs.append(batch_rendered_rgbs.cpu())
                rendered_img = torch.cat(rendered_rgbs, dim=0).view(img.shape)
                metrics.psnr += psnr(img, rendered_img).item()
                metrics.loss /= rays_o.size(0)
                metrics_acc.append(metrics)

                # Save image
                rendered_img = (255. * rendered_img).type(torch.uint8).numpy()
                Image.fromarray(rendered_img).save(cfg.output / f'{name}_{i}.png')
        return metrics_acc

    def loop() -> Tuple[List[TrainMetrics], List[TestMetrics], List[TestMetrics]]:
        train_metrics: List[TrainMetrics] = []
        test_metrics: List[TestMetrics] = []
        train_step = 0
        test_step = 0
        with tqdm(total=steps) as pbar:
            while True:
                for batch in train_loader:
                    renderer.train()

                    if train_step % occupancy_grid_updates == 0:
                        occupancy_grid.update(lambda t: renderer.sigma_decoder(renderer.feature_module(t)))

                    rays_o = batch['rays_o'].to(device)
                    rays_d = batch['rays_d'].to(device)
                    rgbs = batch['rgbs'].to(device)

                    rendered_rgbs, stats = renderer(rays_o, rays_d)
                    loss = loss_fn(rendered_rgbs,rgbs)

                    if cfg.method == 'kplanes':
                        loss += renderer.feature_module.loss_tv() * tv_reg_alpha # type: ignore
                        loss += renderer.feature_module.loss_l1() * l1_reg_alpha # type: ignore

                    optimizer.zero_grad()
                    scaler.scale(loss).backward() # type: ignore
                    optimizer.step()
                    scheduler.step()

                    train_loss = loss.detach().cpu().item()
                    occupancy = occupancy_grid.occupancy()
                    train_metrics.append(TrainMetrics(train_loss, occupancy))
                    pbar.set_postfix(
                        loss=train_loss,
                        occupancy=occupancy,
                        masked_ratio=stats.masked_ratio,
                        rendered_samples= stats.rendered_samples
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