from src.models import VanillaFeatureMLP, VanillaOpacityDecoder, VanillaColorDecoder
from src.models import KPlanesFeatureField, KPlanesExplicitOpacityDecoder, KPlanesExplicitColorDecoder
from src.data import ImagesDataset
from src.core import OccupancyGrid, NerfRenderer, mip360_contract, psnr
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict, cast
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import json
import torch
import torch.amp as amp

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
    train_dataset: Dataset
    test_dataset: Dataset | None
    output: Path
    method: str
    steps: int
    batch_size: int
    eval_every: int
    eval_n : int
    occupancy_res: int

def train_vanilla(cfg: VanillaTrainConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(cfg.train_dataset, cfg.batch_size, shuffle=True)

    feature_module: torch.nn.Module
    sigma_decoder: torch.nn.Module
    rgb_decoder: torch.nn.Module

    if cfg.method == 'vanilla':
        feature_module = VanillaFeatureMLP(10, [256 for k in range(8)])
        dim = feature_module.feature_dim
        sigma_decoder = VanillaOpacityDecoder(dim)
        rgb_decoder = VanillaColorDecoder(4, dim, [128])
    elif cfg.method == 'kplanes':
        feature_module = KPlanesFeatureField(32)
        dim = feature_module.feature_dim
        sigma_decoder = KPlanesExplicitOpacityDecoder(dim)
        rgb_decoder = KPlanesExplicitColorDecoder(dim)
    else:
        raise NotImplementedError(f'Unknown method {cfg.method}.')

    occupancy_grid = OccupancyGrid(cfg.occupancy_res, 1.)

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
        train_step = 0
        test_step = 0
        with tqdm(total=cfg.steps) as pbar:
            while True:
                for batch in train_loader:
                    if train_step >= cfg.steps:
                        return train_metrics, test_metrics
                    renderer.train()

                    rays_o = batch['rays_o'].to(device)
                    rays_d = batch['rays_d'].to(device)
                    rgbs = batch['rgbs'].to(device)

                    if train_step % 32 == 0 and train_step > 0 and False:
                        occupancy_grid.update(lambda t: renderer.sigma_decoder(renderer.feature_module(t)))

                    rendered_rgbs = renderer(rays_o, rays_d)
                    loss = torch.mean((rendered_rgbs - rgbs)**2)

                    if cfg.method == 'kplanes' and False:
                        loss += renderer.feature_module.loss_tv() * 1. # type: ignore
                        loss += renderer.feature_module.loss_l1() * 1e-3 # type: ignore

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss = loss.detach().cpu().item()
                    occupancy = occupancy_grid.grid.sum().item() / occupancy_grid.grid.numel()
                    train_metrics.append(TrainMetrics(train_loss, occupancy))
                    pbar.set_postfix(loss=train_loss, occupancy=occupancy)

                    if cfg.test_dataset and train_step % cfg.eval_every == 0 and train_step > 0:
                        renderer.eval()
                        with torch.no_grad():
                            metrics = TestMetrics()
                            for i in tqdm(range(cfg.eval_n)):
                                data = cfg.test_dataset[test_step % len(cast(ImagesDataset, cfg.test_dataset))]
                                img = data['rgbs']
                                rays_o = data['rays_o'].view(-1, 3)
                                rays_d = data['rays_d'].view(-1, 3)
                                rgbs = data['rgbs'].view(-1, 3)
                                rendered_rgbs = []
                                img_loss = []
                                for k in range(0, len(rays_o), cfg.batch_size):
                                    start = k
                                    end = k + cfg.batch_size
                                    batch_rays_o = rays_o[start:end].to(device)
                                    batch_rays_d = rays_d[start:end].to(device)
                                    batch_rgbs = rgbs[start:end].to(device)

                                    batch_rendered_rgbs = renderer(batch_rays_o, batch_rays_d)
                                    test_loss = torch.mean((batch_rendered_rgbs - batch_rgbs)**2).item()

                                    img_loss.append(test_loss)
                                    rendered_rgbs.append(batch_rendered_rgbs.cpu())
                                rendered_img = torch.cat(rendered_rgbs, dim=0).view(img.shape)
                                metrics.psnr += psnr(img, rendered_img).item()
                                metrics.loss += torch.mean(torch.tensor(img_loss)).item()

                                # Save image
                                rendered_img = (255. * rendered_img).type(torch.uint8).numpy()
                                Image.fromarray(rendered_img).save(cfg.output / f'{train_step}_{i}_test.png')

                                test_step += 1
                            metrics.psnr /= cfg.eval_n
                            metrics.loss /= cfg.eval_n
                            test_metrics.append(metrics)

                    train_step += 1
                    pbar.update(1)

    train_metrics, test_metrics = loop()
    json.dump([asdict(x) for x in train_metrics], open(cfg.output / 'train_metrics.json', 'w'))
    json.dump([asdict(x) for x in test_metrics], open(cfg.output / 'test_metrics.json', 'w'))