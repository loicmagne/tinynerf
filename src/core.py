""" 
we start with a bunch of rays directions and origin of shape [n,3] for which we want to compute the color (i.e. sample along ray + accumulate)

ray marching is simply performed by creating a linspace of timestamps t and adding t * rays_direction to rays_origin
this yields a 2D grid [n, n_samples, 3] for which we can query occupancy grid, and discard samples accordingly
Note: there is no discarding, that would create non rectangular shape, we only keep a mask

we then compute some features for those samples, this gives a [n, n_samples, n_features] tensor

those features are used to compute densities, with those densities we can compute transmittance to perform early ray termination

finally we compute rgbs for remaining samples, and we can accumulate to compute original rgs

+ this method reduces to almost the bare minimum the amount of queries to density and rgbs, and note that sample features and only computed once
+ we always operate on a fixed dimension vector/grid so this should work with triton
- this process consumes more memory than it should, since we over generate samples, if we were to generate samples along ray iteratively, we could avoid
  allocating memory for samples which are discarded by the occupancy grid as done in nerfacc, but that is harder to work with triton
"""

from typing import Callable, List, cast
import numpy as np
import torch
import math

def mip360_contract(coords: torch.Tensor) -> torch.Tensor:
    """Scene contraction from Mip-NeRF 360 https://arxiv.org/abs/2111.12077"""
    norm = torch.norm(coords, float("inf"), -1, keepdim=True) # type: ignore 
    return torch.where(norm <= 1., coords, (2. - 1./norm) * coords / norm) / 2.

class OccupancyGrid(torch.nn.Module):
    def __init__(self, size: List[int] | int):
        super().__init__()
        size = size if isinstance(size, List) else [size, size, size]
        self.n_voxels = math.prod(size)
    
        self.grid: torch.Tensor
        self.register_buffer("grid", torch.zeros(size, dtype=torch.float))
        self.coords = torch.stack(torch.meshgrid([
            torch.arange(size[0], dtype=torch.float),
            torch.arange(size[1], dtype=torch.float),
            torch.arange(size[2], dtype=torch.float)
        ], indexing="ij"), -1).view(-1, 3)
        self.size = torch.tensor(size, dtype=torch.float)

    @torch.no_grad()
    def update(self, occupancy_fn: Callable[[torch.Tensor], torch.Tensor], threshold: float = 0.01):
        coords = (self.coords + torch.rand_like(self.coords)) / self.size # jitter inside voxel 
        coords = coords.to(self.grid.device)
        self.grid = (occupancy_fn(coords) > threshold).view(self.grid.shape).float() # TODO: batch evaluation

    @torch.no_grad()
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """"coords: [..., 3], normalized [-1,1] coordinates"""
        values = torch.nn.functional.grid_sample(
            # self.grid[None,None,...]
            self.grid.unsqueeze(0).unsqueeze(0),
            coords.view(1,-1,1,1,3),
            mode="bilinear", align_corners=False # TODO: align_corners=True?
        ).view(-1)
        return values

def clipped_exponential_stepping(near: float, far: float, delta_min: float, delta_max: float, device: torch.device):
    """Clipped exponential stepping function as described in Instant-NGP paper"""
    t = near
    acc_ts, acc_steps = [t], []
    while t < far:
        step = min(delta_max, max(delta_min, t))
        t += step
        acc_ts.append(t)
        acc_steps.append(step)
    ts = torch.tensor(acc_ts[:-1], device=device)
    steps = torch.tensor(acc_steps, device=device)
    return ts, steps

class NerfRenderer(torch.nn.Module):
    def __init__(
        self,
        occupancy_grid: OccupancyGrid,
        feature_module: torch.nn.Module,
        sigma_decoder: torch.nn.Module,
        rgb_decoder: torch.nn.Module,
        contraction: Callable[[torch.Tensor], torch.Tensor] = mip360_contract,
        near: float = 0.,
        far: float = 1e5,
        delta_min: float = math.sqrt(3.)/1024.,
        delta_max: float = 1e10,
    ):
        super().__init__()
        self.occupancy_grid = occupancy_grid
        self.feature_module = feature_module
        self.sigma_decoder = sigma_decoder
        self.rgb_decoder = rgb_decoder
        self.contraction = contraction
        self.near = near
        self.far = far
        self.delta_min = delta_min
        self.delta_max = delta_max

        assert hasattr(self.feature_module, "feature_dim"), "feature module requires a feature_dim attribute"

    def forward(
        self,
        rays_o: torch.Tensor, # [n, 3]
        rays_d: torch.Tensor, # [n, 3]
        early_termination: float = 1e-4 # early ray termination threshold
    ) -> torch.Tensor:
        device = rays_o.device

        # Generate samples along each ray
        # TODO : precompute and store t_values
        # TODO: distances are wrong for now since they don't account for ray direction norm -> should normalize rays_d?
        t_values, distances = clipped_exponential_stepping(self.near, self.far, self.delta_min, self.delta_max, device)
        # jitter samples along ray when training
        if self.training:
            t_values = t_values + torch.rand_like(t_values) * distances

        n_rays = rays_o.size(0)
        n_samples = t_values.size(0)

        samples_grid = rays_o[:, None, :] + rays_d[:, None, :] * t_values[None, :, None]
        samples_grid = self.contraction(samples_grid)
        mask : torch.Tensor = self.occupancy_grid(samples_grid).view(n_rays, n_samples).bool()

        samples_features = torch.zeros(n_rays, n_samples, cast(int, self.feature_module.feature_dim), device=device)
        samples_sigmas = torch.zeros(n_rays, n_samples, device=device)
        samples_rgbs = torch.zeros(n_rays, n_samples, 3, device=device)

        # compute features and density for remaining samples
        samples_features[mask] = self.feature_module(samples_grid[mask])
        samples_sigmas[mask] = self.sigma_decoder(samples_features[mask]).squeeze()
        
        # compute transmittance and alpha
        alpha = -samples_sigmas * distances[None, :] # not actually alpha
        transmittance = torch.exp(torch.cumsum(alpha, 1))[:, :-1]
        transmittance = torch.cat([torch.ones(n_rays,1).to(device), transmittance], dim=1) # shift transmittance to the right
        alpha = 1. - torch.exp(alpha)
        weights = transmittance * alpha
        
        mask = mask & (weights > early_termination)

        # compute rgb for remaining samples
        samples_rgbs[mask] = self.rgb_decoder(
            samples_features[mask],
            rays_d.unsqueeze(1).expand(-1,n_samples,-1)[mask] # give ray direction to rgb decoder
        ) * weights[mask][:, None]

        rendered_rgb = samples_rgbs.sum(1) # accumulate rgb
        
        return rendered_rgb

def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return - 10. * torch.log10(torch.mean((x - y) ** 2))