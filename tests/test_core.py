import torch
from src.core import OccupancyGrid

def test_occupancy_grid():
    grid = OccupancyGrid(128)
    grid.update(lambda t: 1. * (t[:, 0] > 0.5 ), 0.5)
    
    assert grid.grid.sum().item() >= grid.n_voxels / 3.
    assert grid.grid.sum().item() <= 2. * grid.n_voxels / 3.
    assert grid.grid.size() == (128, 128, 128)