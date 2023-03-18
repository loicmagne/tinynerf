import torch
from src.core import OccupancyGrid, NerfRenderer, RayMarcherAABB, RayMarcherUnbounded, ContractionMip360, ContractionAABB
from src.models import VanillaFeatureMLP, VanillaOpacityDecoder, VanillaColorDecoder

def test_occupancy_grid():
    grid = OccupancyGrid(128)
    grid.grid[:,:,64:] = 0.

    # about 1/2 of the occupancies should be set
    assert grid.grid.sum().item() >= grid.grid.numel() / 3.
    assert grid.grid.sum().item() <= 2. * grid.grid.numel() / 3.
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
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]


    print(grid(unit_coords))
    assert torch.all(grid(unit_coords) == torch.tensor(occs))
 
def test_occupancy_grid_update():
    # setup vanilla nerf
    feature_mlp = VanillaFeatureMLP(10, [256 for k in range(8)])
    opacity_decoder = VanillaOpacityDecoder(256)

    def sigma_fn(t: torch.Tensor):
        features = feature_mlp(t)
        opacity = opacity_decoder(features)
        return opacity

    occupancy_grid = OccupancyGrid(32)
    occupancy_grid.update(sigma_fn)
    assert occupancy_grid.grid.sum().item() <= occupancy_grid.grid.numel()

def test_renderer_vanilla_nerf():
    # setup vanilla nerf
    feature_mlp = VanillaFeatureMLP(10, [256 for k in range(8)])
    opacity_decoder = VanillaOpacityDecoder(256)
    color_decoder = VanillaColorDecoder(4, 256, [128])
    ray_marcher = RayMarcherUnbounded()
    contraction = ContractionMip360()
    
    def occupancy_fn(t: torch.Tensor):
        features = feature_mlp(t)
        opacity = opacity_decoder(features)
        return opacity

    occupancy_grid = OccupancyGrid(64)
    occupancy_grid.update(occupancy_fn)

    renderer  = NerfRenderer(
        occupancy_grid=occupancy_grid,
        feature_module=feature_mlp,
        sigma_decoder=opacity_decoder,
        rgb_decoder=color_decoder,
        contraction=contraction,
        ray_marcher=ray_marcher
    )

    rays_o = torch.rand(100, 3)
    rays_d = torch.rand(100, 3)

    rendered_rgb, _ = renderer(rays_o, rays_d)

    assert rendered_rgb.size() == (100, 3)

def test_ray_marcher_unbounded():
    n_rays = 100
    n_samples = 1000
    rays_o = torch.randn(n_rays, 3)
    rays_d = torch.randn(n_rays, 3)
    ray_marcher = RayMarcherUnbounded(n_samples=n_samples)

    t_values, step_sizes = ray_marcher(rays_o, rays_d)

    assert t_values.size() == (n_rays, n_samples)
    assert step_sizes.size() == (n_rays, n_samples)
    assert torch.all(t_values >= 0.)
    assert torch.all(step_sizes >= 0.)
    
    contraction = ContractionMip360()
    coords = rays_o[:,None,:] + rays_d[:,None,:] * t_values[...,None]
    coords, mask = contraction(coords)

    assert mask is None
    assert torch.all(coords <= 1.)
    assert torch.all(coords >= -1.)

def test_ray_marcher_aabb():
    n_rays = 100
    n_samples = 1000
    aabb = torch.tensor([[0.,0.,0.], [2.,2.,2.]])
    rays_o = torch.randn(n_rays, 3)
    rays_d = torch.randn(n_rays, 3)
    ray_marcher = RayMarcherAABB(aabb=aabb, n_samples=n_samples)

    t_values, step_sizes = ray_marcher(rays_o, rays_d)
    
    assert t_values.size() == (n_rays, n_samples)
    assert step_sizes.size() == (n_rays, n_samples)
    assert torch.all(t_values >= 0.)
    assert torch.all(step_sizes >= 0.)
    
    contraction = ContractionAABB(aabb)
    coords = rays_o[:,None,:] + rays_d[:,None,:] * t_values[...,None]
    _, mask = contraction(coords)
    print(mask.size())
    print(coords.size())
    print(coords[mask].size())
    print((coords[mask] >= aabb[0]).size())
    print((coords[mask] <= aabb[1]).size())
    print((coords[mask] >= aabb[0]) & (coords[mask] <= aabb[1]))
    assert torch.all((coords[mask] >= aabb[0]) & (coords[mask] <= aabb[1]))