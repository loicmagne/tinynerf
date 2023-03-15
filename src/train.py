from src.data import NerfDataset, NerfData, parse_nerf_synthetic
from src.models import VanillaFeatureMLP, VanillaOpacityDecoder, VanillaColorDecoder
from src.core import OccupancyGrid, NerfRenderer, mip360_contract, psnr
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict
from PIL import Image
from pathlib import Path
import json
import torch

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
    train_dataset: NerfDataset
    test_dataset: NerfDataset | None
    output: Path
    steps: int
    batch_size: int
    eval_every: int

def train_vanilla(cfg: VanillaTrainConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(cfg.train_dataset, cfg.batch_size, shuffle=True)
    test_loader = DataLoader(cfg.test_dataset, cfg.batch_size, shuffle=False) if cfg.test_dataset else None
    
    feature_module = VanillaFeatureMLP(10, [256 for k in range(8)])
    sigma_decoder = VanillaOpacityDecoder(feature_module.feature_dim)
    rgb_decoder = VanillaColorDecoder(4, feature_module.feature_dim, [128])
    
    occupancy_fn = lambda t: sigma_decoder(feature_module(t))
    occupancy_grid = OccupancyGrid(32)

    renderer = NerfRenderer(
        occupancy_grid,
        feature_module,
        sigma_decoder,
        rgb_decoder,
        mip360_contract,
        near=0.,
        far=1e10,
    ).to(device)
    optimizer = torch.optim.Adam(renderer.parameters(), lr=3e-4)

    def loop() -> tuple[list[TrainMetrics], list[TestMetrics]]: 
        train_metrics: List[TrainMetrics] = []
        test_metrics: List[TestMetrics] = []
        step = 0
        with tqdm(total=cfg.steps) as pbar:
            while True:
                for batch in train_loader:
                    if step >= cfg.steps:
                        return train_metrics, test_metrics
                    renderer.train()

                    rays_o = batch['rays_o'].to(device)
                    rays_d = batch['rays_d'].to(device)
                    rgbs = batch['rgbs'].to(device)

                    if step < 256 or step % 32 == 0:
                        occupancy_grid.update(occupancy_fn, 0.01)

                    rendered_rgbs = renderer(rays_o, rays_d)
                    loss = torch.mean((rendered_rgbs - rgbs)**2)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss = loss.detach().cpu().item()
                    occupancy = occupancy_grid.grid.sum().item() / occupancy_grid.grid.numel()
                    train_metrics.append(TrainMetrics(train_loss, occupancy))
                    pbar.set_postfix(loss=train_loss, occupancy=occupancy)
                
                    if test_loader and cfg.test_dataset and step % cfg.eval_every == 0 and step > 0:
                        renderer.eval()
                        with torch.no_grad():
                            acc_rendered_rgbs, acc_rgbs, acc_indices = [], [], []
                            metrics = TestMetrics()
                            for batch in test_loader:
                                rays_o = batch['rays_o'].to(device)
                                rays_d = batch['rays_d'].to(device)
                                rgbs = batch['rgbs'].to(device)
                                indices = batch['indices']

                                rendered_rgbs = renderer(rays_o, rays_d)
                                loss = torch.mean((rendered_rgbs - rgbs)**2)

                                test_loss = loss.item()
                                metrics.loss += test_loss

                                acc_rendered_rgbs.append(rendered_rgbs.cpu())
                                acc_rgbs.append(rgbs.cpu())
                                acc_indices.append(indices.cpu())

                            original_images = cfg.test_dataset.recover_images(
                                torch.cat(acc_rgbs, dim=0),
                                torch.cat(acc_indices, dim=0)
                            )
                            rendered_images = cfg.test_dataset.recover_images(
                                torch.cat(acc_rendered_rgbs, dim=0),
                                torch.cat(acc_indices, dim=0)
                            )

                            psnrs = [psnr(original_images[i], rendered_images[i]) for i in range(len(original_images))]
                            metrics.psnr = torch.mean(torch.tensor(psnrs)).item()
                            metrics.loss /= len(test_loader)
                            test_metrics.append(metrics)
 
                            for i, img in enumerate(rendered_images):
                                img = (255. * img.permute(1, 2, 0)).type(torch.uint8).numpy()
                                Image.fromarray(img).save(cfg.output / f'{step}_{i}_test.png')

                    step += 1
                    pbar.update(1)

    train_metrics, test_metrics = loop()
    json.dump([asdict(x) for x in train_metrics], open(cfg.output / 'train_metrics.json', 'w'))
    json.dump([asdict(x) for x in test_metrics], open(cfg.output / 'test_metrics.json', 'w'))