from typing import cast
from pathlib import Path
from src.data import parse_nerf_synthetic, NerfDataset

def test_synthetic_dataset():
    data = parse_nerf_synthetic(Path("tests/dummy/hotdog"), "train")
    dataset = NerfDataset(data)
    assert dataset.rays_d.size() == dataset.rays_o.size()
    assert dataset.indices.size(0) == dataset.rays_d.size(0)
    assert dataset.rgbs is not None and dataset.rgbs.size(0) == dataset.rays_d.size(0)
