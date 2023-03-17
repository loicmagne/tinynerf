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

from typing import Callable, Tuple, List, cast
from dataclasses import dataclass
import torch

def mip360_contract(coords: torch.Tensor) -> torch.Tensor:
    """Scene contraction from Mip-NeRF 360 https://arxiv.org/abs/2111.12077"""
    norm = torch.norm(coords, dim=-1, keepdim=True) # type: ignore 
    return torch.where(norm <= 1., coords, (2. - 1./norm) * coords / norm) / 2.

def unbounded_stepping(near: float, uniform_range: float, n_samples: int, device: torch.device):
    """uniform steps between near and near+uniform_range, then uniform disparity steps
    n_samples//2 samples are used for uniform range, n_samples//2 for disparity range"""
    ts = torch.linspace(0., 1.-(1./(n_samples+2)), n_samples+1, device=device)
    f = lambda x: torch.where(x < 0.5, 2 * x, 1 / (2 - 2 * x))
    ts = f(ts) * uniform_range + near
    steps = ts[1:] - ts[:-1]
    return ts[:-1], steps

def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return - 10. * torch.log10(torch.mean((x - y) ** 2))

class OccupancyGrid(torch.nn.Module):
    def __init__(self, size: List[int] | int, decay: float = 0.95):
        super().__init__()
        size = size if isinstance(size, List) else [size, size, size]
        self.decay = decay
    
        self.grid: torch.Tensor
        self.register_buffer("grid", torch.ones(size, dtype=torch.float))
        self.coords = torch.stack(torch.meshgrid([
            torch.arange(size[0], dtype=torch.float),
            torch.arange(size[1], dtype=torch.float),
            torch.arange(size[2], dtype=torch.float)
        ], indexing="ij"), -1)
        self.size = torch.tensor(size, dtype=torch.float)

    @torch.no_grad()
    def update(self, sigma_fn: Callable[[torch.Tensor], torch.Tensor]):
        batch_shape = self.grid[0].size()
        for i in range(self.grid.size(0)):
            coords = (self.coords[i] + torch.rand_like(self.coords[i])) / self.size # jitter inside voxel 
            coords = coords.view(-1, 3).contiguous().to(self.grid.device)
            alpha = (1. - torch.exp(-sigma_fn(coords) * 1.)).view(batch_shape) # TODO: add step size
            self.grid[i] = torch.maximum(self.grid[i] * self.decay, alpha)

    @torch.no_grad()
    def forward(self, coords: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
        """"coords: [..., 3], normalized [-1,1] coordinates"""
        new_shape = coords.shape[:-1]
        values = torch.nn.functional.grid_sample(
            # self.grid[None,None,...]
            self.grid.unsqueeze(0).unsqueeze(0),
            coords.view(1,-1,1,1,3),
            mode="bilinear", align_corners=False # TODO: align_corners=True?
        ).view(new_shape).contiguous()
        return values > threshold

@dataclass
class RenderingStats:
    total_samples: int = 0
    skipped_opacity: float = 0.
    skipped_transmittance: float = 0.

class NerfRenderer(torch.nn.Module):
    def __init__(
        self,
        occupancy_grid: OccupancyGrid,
        feature_module: torch.nn.Module,
        sigma_decoder: torch.nn.Module,
        rgb_decoder: torch.nn.Module,
        contraction: Callable[[torch.Tensor], torch.Tensor] = mip360_contract,
        near: float = 0.,
        scene_scale: float = 1.,
        delta_min: float = 1e-4,
        delta_max: float = 1e10,
    ):
        super().__init__()
        self.occupancy_grid = occupancy_grid
        self.feature_module = feature_module
        self.sigma_decoder = sigma_decoder
        self.rgb_decoder = rgb_decoder
        self.contraction = contraction
        self.near = near
        self.scene_scale = scene_scale
        self.delta_min = delta_min
        self.delta_max = delta_max

        assert hasattr(self.feature_module, "feature_dim"), "feature module requires a feature_dim attribute"

    def forward(
        self,
        rays_o: torch.Tensor, # [n, 3]
        rays_d: torch.Tensor, # [n, 3]
        n_samples: int = 250,
        opacity_threshold: float = 1e-2,
        early_termination_threshold: float = 1e-4
    ) -> Tuple[torch.Tensor, RenderingStats]:
        device = rays_o.device
        n_rays = rays_o.size(0)

        stats = RenderingStats()
        stats.total_samples = n_rays * n_samples

        # Generate samples along each ray
        # TODO : precompute and store t_values
        t_values, distances = unbounded_stepping(self.near, self.scene_scale, n_samples, device)
        # jitter samples along ray when training
        t_values = t_values[None,:].expand(n_rays, -1) # expand to [n_rays, n_samples] so we can jitter differently for each ray
        distances = distances[None,:].expand(n_rays, -1)
        if self.training:
            t_values = t_values + torch.rand_like(t_values) * distances

        samples = rays_o[:,None,:] + rays_d[:,None,:] * t_values[...,None]
        samples = self.contraction(samples)

        mask : torch.Tensor = self.occupancy_grid(samples, threshold=opacity_threshold)
        stats.skipped_opacity = 1. - mask.float().mean().item()

        samples_features = torch.zeros(n_rays, n_samples, cast(int, self.feature_module.feature_dim), device=device)
        samples_sigmas = torch.zeros(n_rays, n_samples, device=device)
        samples_rgbs = torch.zeros(n_rays, n_samples, 3, device=device)

        # compute features and density for remaining samples
        samples_features[mask] = self.feature_module(samples[mask])
        samples_sigmas[mask] = self.sigma_decoder(samples_features[mask]).squeeze()
 
        # compute transmittance and alpha
        alpha = -samples_sigmas * distances # not actually alpha
        transmittance = torch.exp(torch.cumsum(alpha, 1))[:, :-1]
        transmittance = torch.cat([torch.ones(n_rays,1).to(device), transmittance], dim=1) # shift transmittance to the right
        alpha = 1. - torch.exp(alpha)
        weights = transmittance * alpha

        mask = mask & (transmittance > early_termination_threshold)
        stats.skipped_transmittance = 1. - mask.float().mean().item()

        # compute rgb for remaining samples
        samples_rgbs[mask] = self.rgb_decoder(
            samples_features[mask],
            rays_d.unsqueeze(1).expand(-1,n_samples,-1)[mask] # give ray direction to rgb decoder
        ) * weights[mask][:, None]

        rendered_rgb = samples_rgbs.sum(1) # accumulate rgb

        return rendered_rgb, stats