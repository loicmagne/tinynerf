from typing import cast
from pathlib import Path
from src.data import parse_nerf_synthetic, RaysDataset, ImagesDataset
from torch.utils.data import DataLoader
import torch

def test_synthetic_dataset():
    data = parse_nerf_synthetic(Path("tests/dummy/hotdog"), "train")
    ray_dataset = RaysDataset(data)
    img_dataset = ImagesDataset(data)

    assert len(img_dataset.rays_d) == len(img_dataset.rays_o)
    assert img_dataset.rgbs is not None and len(img_dataset.rgbs) == len(img_dataset.rays_d)

    assert ray_dataset.rays_d.size() == ray_dataset.rays_o.size()
    assert ray_dataset.rgbs is not None and ray_dataset.rgbs.size(0) == ray_dataset.rays_d.size(0)
    assert ray_dataset.rgbs.min() >= 0. and ray_dataset.rgbs.max() <= 1.
    assert ray_dataset.rgbs.dtype == torch.float
    assert ray_dataset.rays_d.dtype == torch.float
    assert ray_dataset.rays_o.dtype == torch.float

    loader = DataLoader(ray_dataset, batch_size=1024, shuffle=False)
    rgbs = []
    for batch in loader:
        rgbs.append(batch['rgbs'])
    rgbs = torch.cat(rgbs, dim=0)
    assert rgbs.size(0) == ray_dataset.rays_d.size(0)