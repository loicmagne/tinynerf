from src.data import NerfDataset, NerfData, parse_nerf_synthetic
from src.models import VanillaFeatureMLP, VanillaOpacityDecoder, VanillaColorDecoder
from src.core import OccupancyGrid, NerfRenderer, mip360_contract, psnr
from dataclasses import dataclass
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict
import torch

@dataclass
class StepMetrics:
    loss: float = 0.

@dataclass
class EpochMetrics:
    train_loss: float = 0.
    test_loss: float = 0.
    psnr: float = 0.
    ssim: float = 0.

@dataclass
class VanillaTrainConfig:
    train_dataset: NerfDataset
    test_dataset: NerfDataset | None
    steps: int
    batch_size: int
    eval_every: int

def train_vanilla(cfg: VanillaTrainConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(cfg.train_dataset, cfg.batch_size, shuffle=True)
    test_loader = DataLoader(cfg.test_dataset, cfg.batch_size, shuffle=False) if cfg.test_dataset else None
    
    feature_module = VanillaFeatureMLP(10, [256 for k in range(8)])
    sigma_decoder = VanillaOpacityDecoder(256)
    rgb_decoder = VanillaColorDecoder(4, 256, [128])
    
    occupancy_fn = lambda t: sigma_decoder(feature_module(t))
    occupancy_grid = OccupancyGrid(128)

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

    # TODO: DEVICES!!
    metrics : Dict[str, List]= { 'steps': [], 'epochs': [] }

    epochs = cfg.steps // len(train_loader) + 1
    step = 0
    for epoch in range(epochs):
        epoch_metrics = EpochMetrics()

        renderer.train()
        with tqdm(train_loader) as loader:
            for batch in loader:
                step_metrics = StepMetrics()

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
                step_metrics.loss = train_loss
                epoch_metrics.train_loss += train_loss

                loader.set_postfix(loss=train_loss)
                metrics['steps'].append(step_metrics)
                step += 1
        
        if test_loader and epoch % cfg.eval_every == 0 and epoch > 0:
            renderer.eval()
            with torch.no_grad():
                all_rendered_rgbs, all_rgbs, all_indices = [], [], []
                for batch in test_loader:
                    rays_o = batch['rays_o'].to(device)
                    rays_d = batch['rays_d'].to(device)
                    rgbs = batch['rgbs'].to(device)
                    indices = batch['indices']

                    rendered_rgbs = renderer(rays_o, rays_d)
                    loss = torch.mean((rendered_rgbs - rgbs)**2)

                    test_loss = loss.detach().cpu().item()
                    epoch_metrics.test_loss += test_loss

                    all_rendered_rgbs.append(rendered_rgbs.cpu())
                    all_rgbs.append(rgbs.cpu())
                    all_indices.append(indices.cpu())
                original_images = cfg.test_dataset.recover_images(torch.cat(rgbs, dim=0), torch.cat(indices, dim=0)) # type: ignore
                rendered_images = cfg.test_dataset.recover_images(torch.cat(rendered_rgbs, dim=0), torch.cat(indices, dim=0)) # type: ignore

                psnrs = [psnr(original_images[i], rendered_images[i]) for i in range(len(original_images))]
                epoch_metrics.psnr = torch.mean(torch.tensor(psnrs)).cpu().item()
                
                epoch_metrics.test_loss /= len(test_loader)
        epoch_metrics.train_loss /= len(train_loader)
        metrics['epochs'].append(epoch_metrics)