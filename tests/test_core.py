import torch
from src.core import OccupancyGrid, NerfRenderer, mip360_contract, clipped_exponential_stepping
from src.models import VanillaFeatureMLP, VanillaOpacityDecoder, VanillaColorDecoder

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
    
def test_occupancy_grid_update():
    # setup vanilla nerf
    feature_mlp = VanillaFeatureMLP(10, [256 for k in range(8)])
    opacity_decoder = VanillaOpacityDecoder(256)

    def occupancy_fn(t: torch.Tensor):
        features = feature_mlp(t)
        opacity = opacity_decoder(features)
        return opacity

    occupancy_grid = OccupancyGrid(32)
    occupancy_grid.update(occupancy_fn, 0.)
    assert occupancy_grid.grid.sum().item() == occupancy_grid.n_voxels

    occupancy_grid.update(occupancy_fn, 0.5)
    assert occupancy_grid.grid.sum().item() <= occupancy_grid.n_voxels

def test_renderer_vanilla_nerf():
    # setup vanilla nerf
    feature_mlp = VanillaFeatureMLP(10, [256 for k in range(8)])
    opacity_decoder = VanillaOpacityDecoder(256)
    color_decoder = VanillaColorDecoder(4, 256, [128])
    
    def occupancy_fn(t: torch.Tensor):
        features = feature_mlp(t)
        opacity = opacity_decoder(features)
        return opacity

    occupancy_grid = OccupancyGrid(64)
    occupancy_grid.update(occupancy_fn, 0.001)

    renderer = NerfRenderer(
        occupancy_grid,
        feature_mlp,
        opacity_decoder,
        color_decoder,
        mip360_contract
    )

    rays_o = torch.rand(100, 3)
    rays_d = torch.rand(100, 3)

    rendered_rgb = renderer(rays_o, rays_d)

    assert rendered_rgb.size() == (100, 3)

def test_stepping():
    t_values, distances = clipped_exponential_stepping(0., 1e5, 1e-3, 1e10, torch.device('cpu'))
    assert torch.all(t_values[1:] > t_values[:-1])
    assert torch.all(t_values <= 1e5)
    assert torch.all(t_values >= 0.)
    assert torch.all(t_values[:-1] + distances[:-1] == t_values[1:])