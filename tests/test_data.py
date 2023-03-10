from src.data import parse_nerf_synthetic, NerfDataset
from pathlib import Path

def test_synthetic_dataset():
    data = parse_nerf_synthetic(Path("tests/dummy/hotdog"), "train")
    dataset = NerfDataset(data)