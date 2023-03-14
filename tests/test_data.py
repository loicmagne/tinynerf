from typing import cast
from pathlib import Path
from src.data import parse_nerf_synthetic, NerfDataset
from torch.utils.data import DataLoader
import torch

def test_synthetic_dataset():
    data = parse_nerf_synthetic(Path("tests/dummy/hotdog"), "train")
    dataset = NerfDataset(data)
    assert dataset.rays_d.size() == dataset.rays_o.size()
    assert dataset.indices.size(0) == dataset.rays_d.size(0)
    assert dataset.rgbs is not None and dataset.rgbs.size(0) == dataset.rays_d.size(0)
    assert dataset.rgbs.min() >= 0. and dataset.rgbs.max() <= 1.
    assert dataset.rgbs.dtype == torch.float
    assert dataset.rays_d.dtype == torch.float
    assert dataset.rays_o.dtype == torch.float

    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    rgbs, indices = [], []
    for batch in loader:
        rgbs.append(batch['rgbs'])
        indices.append(batch['indices'])
    rgbs = torch.cat(rgbs, dim=0)
    indices = torch.cat(indices, dim=0)
    reconstructed_imgs = dataset.recover_images(rgbs, indices)
    assert rgbs.size(0) == dataset.rays_d.size(0)
    assert len(reconstructed_imgs) == 2
    assert reconstructed_imgs[0].size(1) == dataset.shape[0][0]
    assert reconstructed_imgs[0].size(2) == dataset.shape[0][1]
    assert reconstructed_imgs[1].size(1) == dataset.shape[1][0]
    assert reconstructed_imgs[1].size(2) == dataset.shape[1][1]