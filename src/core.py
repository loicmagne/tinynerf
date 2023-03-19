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
from functools import cached_property
import torch


@dataclass
class ContractionMip360():
    order : float | int = float('inf')

    @torch.no_grad()
    def __call__(self, coords: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Scene contraction from Mip-NeRF 360 https://arxiv.org/abs/2111.12077"""
        norm = torch.norm(coords, p=self.order, dim=-1, keepdim=True) # type: ignore 
        coords = torch.where(norm <= 1., coords, (2. - 1./norm) * coords / norm) / 2.
        return coords, None

@dataclass
class ContractionAABB():
    aabb: torch.Tensor # [2,3]

    @torch.no_grad()
    def __call__(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Contract coords inside aabb to [-1,1], aabb.shape = [2,3]"""
        mask = torch.all((coords >= self.aabb[0]) & (coords <= self.aabb[1]), dim=-1)
        coords = (coords - self.aabb[0]) / (self.aabb[1] - self.aabb[0]) * 2. - 1.
        return coords, mask

Contraction = ContractionMip360 | ContractionAABB

@dataclass
class RayMarcherUnbounded():
    n_samples: int = 200
    near: float = 0.
    far: float = 1e5
    uniform_range: float = 1.

    @cached_property
    def step_size(self) -> float:
        return self.uniform_range / self.n_samples

    @torch.no_grad()
    def __call__(self, rays_o: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = rays_o.device
        n_rays = rays_o.size(0)

        f = lambda x: torch.where(x < 0.5, 2 * x, 1 / (2 - 2 * x))
        t_values = torch.linspace(0., 1.-(1./(self.n_samples+2)), self.n_samples+1, device=device)
        t_values  = f(t_values ) * self.uniform_range + self.near
        step_sizes = t_values[1:] - t_values[:-1]

        t_values = torch.broadcast_to(t_values[:-1], (n_rays, self.n_samples))
        step_sizes = torch.broadcast_to(step_sizes, (n_rays, self.n_samples))
        return t_values, step_sizes


@dataclass
class RayMarcherAABB():
    aabb: torch.Tensor
    n_samples: int = 200
    near: float = 0.
    far: float = 1e5

    @cached_property
    def step_size(self) -> float:
        return torch.norm(self.aabb[1] - self.aabb[0]) / self.n_samples

    @torch.no_grad()
    def __call__(self, rays_o: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = rays_o.device
        eps = 1e-9

        # Compute rays intersections with bounding box
        aabb_distances = self.aabb.unsqueeze(1) - rays_o
        aabb_intersections = aabb_distances / torch.where(rays_d == 0., rays_d + eps, rays_d)
        t_min = torch.amax(torch.amin(aabb_intersections, dim=0), dim=1)
        t_max = torch.amin(torch.amax(aabb_intersections, dim=0), dim=1)
        t_min = torch.clamp(t_min, min=self.near, max=self.far) # !!!!!!!!!!!!

        # Compute samples along valid rays
        step_size: float = torch.norm(self.aabb[1] - self.aabb[0]) / self.n_samples
        steps = torch.arange(self.n_samples, dtype=torch.float, device=device) * step_size
        t_values = t_min[:, None] + steps
        step_sizes = torch.full_like(t_values, step_size)

        return t_values, step_sizes

RayMarcher = RayMarcherUnbounded | RayMarcherAABB


class OccupancyGrid(torch.nn.Module):
    def __init__(
        self,
        size: List[int] | int,
        step_size: float,
        threshold: float = 0.01,
        decay: float = 0.95,
    ):
        super().__init__()
        size = size if isinstance(size, List) else [size, size, size]
        self.decay = decay
        self.step_size = step_size
        self.threshold = threshold
    
        self.grid: torch.Tensor
        self.register_buffer("grid", torch.zeros(size, dtype=torch.float))
        self.coords = torch.stack(torch.meshgrid([
            torch.arange(size[0], dtype=torch.float),
            torch.arange(size[1], dtype=torch.float),
            torch.arange(size[2], dtype=torch.float)
        ], indexing="ij"), -1)
        self.size = torch.tensor(size, dtype=torch.float)
        self.mean = 1.

    @torch.no_grad()
    def occupancy(self) -> float:
        return (self.grid > self.threshold).sum().item() / self.grid.numel()

    @torch.no_grad()
    def update(self, sigma_fn: Callable[[torch.Tensor], torch.Tensor]):
        batch_shape = self.grid[0].size()
        for i in range(self.grid.size(0)):
            coords = -1. + 2. * (self.coords[i] + torch.rand_like(self.coords[i])) / self.size # jitter inside voxel 
            coords = coords.view(-1, 3).contiguous().to(self.grid.device)
            alpha = (1. - torch.exp(-sigma_fn(coords) * self.step_size)).view(batch_shape) # TODO: add step size
            self.grid[i] = torch.maximum(self.grid[i] * self.decay, alpha)
        self.mean = self.grid.mean().item()

    @torch.no_grad()
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """"coords: [..., 3], normalized [-1,1] coordinates"""
        new_shape = coords.shape[:-1]
        values = torch.nn.functional.grid_sample(
            # self.grid[None,None,...]
            self.grid.unsqueeze(0).unsqueeze(0),
            coords.view(1,-1,1,1,3),
            mode="bilinear", align_corners=False # TODO: align_corners=True?
        ).view(new_shape).contiguous()
        return values > min(self.threshold, self.mean)

@dataclass
class RenderingStats:
    skipped_samples: List[float]

class NerfRenderer(torch.nn.Module):
    def __init__(
        self,
        occupancy_grid: OccupancyGrid,
        feature_module: torch.nn.Module,
        sigma_decoder: torch.nn.Module,
        rgb_decoder: torch.nn.Module,
        contraction: Contraction,
        ray_marcher: RayMarcher,
        bg_color: torch.Tensor | None = None,
    ):
        super().__init__()
        self.occupancy_grid = occupancy_grid
        self.feature_module = feature_module
        self.sigma_decoder = sigma_decoder
        self.rgb_decoder = rgb_decoder
        self.contraction = contraction
        self.ray_marcher = ray_marcher
        self.bg_color = bg_color

        assert hasattr(self.feature_module, "feature_dim"), "feature module requires a feature_dim attribute"

    def forward(
        self,
        rays_o: torch.Tensor, # [n, 3]
        rays_d: torch.Tensor, # [n, 3]
        early_termination_threshold: float = 1e-4,
    ) -> Tuple[torch.Tensor, RenderingStats]:
        device = rays_o.device
        n_rays = rays_o.size(0)
        n_samples = self.ray_marcher.n_samples
        stats = RenderingStats(skipped_samples=[])

        # Generate samples along each ray
        # TODO : precompute and store t_values
        t_values, step_sizes = self.ray_marcher(rays_o, rays_d)
        if self.training: # jitter samples along ray when training
            t_values = t_values + torch.rand_like(t_values) * step_sizes # TODO: jittering can get samples out of AABB
        samples = rays_o[:,None,:] + rays_d[:,None,:] * t_values[...,None]
        samples, init_mask = self.contraction(samples)
        mask = init_mask if init_mask is not None else torch.ones((n_rays,n_samples), dtype=torch.bool, device=device)
        stats.skipped_samples.append(1. - mask.float().mean().item())

        mask = mask & self.occupancy_grid(samples)
        stats.skipped_samples.append(1. - mask.float().mean().item())

        samples_features = torch.zeros(n_rays, n_samples, cast(int, self.feature_module.feature_dim), device=device)
        samples_sigmas = torch.zeros(n_rays, n_samples, device=device)
        samples_rgbs = torch.zeros(n_rays, n_samples, 3, device=device)

        # compute features and density for remaining samples
        samples_features[mask] = self.feature_module(samples[mask])
        samples_sigmas[mask] = self.sigma_decoder(samples_features[mask]).squeeze()

        # compute transmittance and alpha
        alpha = - samples_sigmas * step_sizes # not actually alpha
        transmittance = torch.exp(torch.cumsum(alpha, 1))[:, :-1]
        transmittance = torch.cat([torch.ones(n_rays,1).to(device), transmittance], dim=1) # shift transmittance to the right
        alpha = 1. - torch.exp(alpha)
        weights = transmittance * alpha

        mask = mask & (transmittance > early_termination_threshold)
        stats.skipped_samples.append(1. - mask.float().mean().item())

        # compute rgb for remaining samples
        samples_rgbs[mask] = self.rgb_decoder(
            samples_features[mask],
            rays_d.unsqueeze(1).expand(-1,n_samples,-1)[mask] # give ray direction to rgb decoder
        ) * weights[mask][:, None]

        rendered_rgb = samples_rgbs.sum(1) # accumulate rgb

        if self.bg_color is not None:
            opacities = weights.sum(1, keepdim=True) # [n_rays, 1]
            # TODO: learn a background color ?
            rendered_rgb = rendered_rgb + self.bg_color.to(device) * (1 - opacities)

        return rendered_rgb, stats