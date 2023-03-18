from src.models import VanillaFeatureMLP, VanillaOpacityDecoder, VanillaColorDecoder
from src.models import KPlanesFeatureField, KPlanesExplicitOpacityDecoder, KPlanesExplicitColorDecoder
from src.data import ImagesDataset, RaysDataset
from src.core import OccupancyGrid, NerfRenderer, mip360_contract, psnr
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, cast
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
    train_rays: RaysDataset
    train_images: ImagesDataset
    test_images: ImagesDataset
    output: Path
    method: str
    steps: int
    batch_size: int
    eval_every: int
    eval_n : int
    occupancy_res: int

def train_vanilla(cfg: VanillaTrainConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(cfg.train_rays, cfg.batch_size, shuffle=True)

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

    occupancy_grid = OccupancyGrid(cfg.occupancy_res, 0.95)

    renderer = NerfRenderer(
        occupancy_grid=occupancy_grid,
        feature_module=feature_module,
        sigma_decoder=sigma_decoder,
        rgb_decoder=rgb_decoder,
        contraction=mip360_contract,
        near=0.1,
        scene_scale=cfg.train_rays.scene_scale/2.,
    ).to(device)
    optimizer = torch.optim.Adam(renderer.parameters(), lr=3e-4)

    def eval(img_dataset: ImagesDataset, n: int, name: str):
        count: int = 0
        def closure(step: int):
            nonlocal count
            renderer.eval()
            metrics_acc = []
            with torch.no_grad():
                metrics = TestMetrics()
                for i in tqdm(range(n)):
                    data = img_dataset[count % len(cast(ImagesDataset, img_dataset))]
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

                        batch_rendered_rgbs, _ = renderer(batch_rays_o, batch_rays_d)
                        test_loss = torch.mean((batch_rendered_rgbs - batch_rgbs)**2).item()

                        img_loss.append(test_loss)
                        rendered_rgbs.append(batch_rendered_rgbs.cpu())
                    rendered_img = torch.cat(rendered_rgbs, dim=0).view(img.shape)
                    metrics.psnr += psnr(img, rendered_img).item()
                    metrics.loss += torch.mean(torch.tensor(img_loss)).item()

                    # Save image
                    rendered_img = (255. * rendered_img).type(torch.uint8).numpy()
                    Image.fromarray(rendered_img).save(cfg.output / f'{step}_{count}_{name}.png')

                    count += 1
                metrics.psnr /= cfg.eval_n
                metrics.loss /= cfg.eval_n
                metrics_acc.append(metrics)
            return metrics_acc
        return closure
    
    train_eval = eval(cfg.train_images, cfg.eval_n, 'train')
    test_eval = eval(cfg.test_images, cfg.eval_n, 'test')

    def loop() -> tuple[list[TrainMetrics], list[TestMetrics]]: 
        train_metrics: List[TrainMetrics] = []
        test_metrics: List[TestMetrics] = []
        train_step = 0
        loss_fn = torch.nn.MSELoss()
        with tqdm(total=cfg.steps) as pbar:
            while True:
                for batch in train_loader:
                    if train_step >= cfg.steps:
                        return train_metrics, test_metrics
                    renderer.train()

                    rays_o = batch['rays_o'].to(device)
                    rays_d = batch['rays_d'].to(device)
                    rgbs = batch['rgbs'].to(device)

                    if train_step % 16 == 0 or train_step < 256:
                        occupancy_grid.update(lambda t: renderer.sigma_decoder(renderer.feature_module(t)))

                    rendered_rgbs, stats = renderer(rays_o, rays_d)
                    loss = loss_fn(rendered_rgbs,rgbs)

                    if cfg.method == 'kplanes':
                        loss += renderer.feature_module.loss_tv() * 0.0001 # type: ignore
                        # loss += renderer.feature_module.loss_l1() * 1e-3 # type: ignore

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss = loss.detach().cpu().item()
                    occupancy = occupancy_grid.grid.sum().item() / occupancy_grid.grid.numel()
                    train_metrics.append(TrainMetrics(train_loss, occupancy))
                    pbar.set_postfix(loss=train_loss, occupancy=occupancy, **asdict(stats))

                    if train_step % cfg.eval_every == 0 and train_step > 0:
                        # train_eval(train_step)
                        metrics = test_eval(train_step)
                        test_metrics.extend(metrics)

                    train_step += 1
                    pbar.update(1)

    train_metrics, test_metrics = loop()
    json.dump([asdict(x) for x in train_metrics], open(cfg.output / 'train_metrics.json', 'w'))
    json.dump([asdict(x) for x in test_metrics], open(cfg.output / 'test_metrics.json', 'w'))