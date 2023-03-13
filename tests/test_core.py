import torch
from src.core import OccupancyGrid

def test_occupancy_grid():
    grid = OccupancyGrid(128)
    def occupancy_fn(t: torch.Tensor):
        return 1. * (torch.min(t[:,:2], -1)[0] > 0.5)
    grid.update(occupancy_fn, 0.5)
    
    # about 1/4 of the occupancies should be set
    assert grid.grid.sum().item() >= grid.n_voxels / 5.
    assert grid.grid.sum().item() <= 2. * grid.n_voxels / 5.
    assert grid.grid.size() == (128, 128, 128)

    coords = [ 
        [32,32,32],
        [32,32,96],
        [32,96,32],
        [32,96,96],
        [96,32,32],
        [96,32,96],
        [96,96,32],
        [96,96,96],
    ]
    unit_coords = 2. * (torch.tensor(coords) / grid.size) - 1.
    occs = [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True
    ]
    for c, o in zip(coords, occs):
        assert grid.grid[c[2], c[1], c[0]] == o
    
    assert torch.all(grid(unit_coords) == torch.tensor(occs))