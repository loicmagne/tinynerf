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

from typing import Callable, Tuple, List, Any
from dataclasses import dataclass
from functools import cached_property
from torch.utils.cpp_extension import load
import torch

_cuda = load(name="_cuda", sources=['src/cuda.cu'], verbose=True)

"""UTILS: Scene contraction, Ray marching strategies, and Occupancy grid"""

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
        t_min = torch.clamp(t_min, min=self.near, max=self.far) # !!!!!!!!!!!!

        # Compute samples along valid rays
        steps = torch.arange(self.n_samples, dtype=torch.float, device=device) * self.step_size
        t_values = t_min[:, None] + steps
        step_sizes = torch.full_like(t_values, self.step_size)

        return t_values, step_sizes

RayMarcher = RayMarcherUnbounded | RayMarcherAABB


class OccupancyGrid(torch.nn.Module):
    def __init__(
        self,
        size: List[int] | int, # !!! size is [depth, height, width]
        step_size: float,
        threshold: float = 0.01,
        decay: float = 0.95,
    ):
        super().__init__()
        size = size if isinstance(size, List) else [size, size, size]
        self.decay = decay
        self.step_size = step_size
        self.base_threshold = threshold

        self.grid: torch.Tensor
        self.register_buffer("grid", torch.ones(size, dtype=torch.float))
        self.size = torch.tensor(size, dtype=torch.float)
        self.mean = 1.

        self.coords = torch.stack(torch.meshgrid([
            torch.arange(size[0], dtype=torch.float),
            torch.arange(size[1], dtype=torch.float),
            torch.arange(size[2], dtype=torch.float)
        ], indexing="ij"), -1)
        # since grid is stored in [depth, height, width] order, the real words coordinate corresponding
        # the voxel addressed by grid[z,y,x] corresponds to coordinates [x,y,z], so we need to flip the coords
        self.coords = torch.flip(self.coords, [-1])

    @torch.no_grad()
    def occupancy(self) -> float:
        return (self.grid > self.threshold).sum().item() / self.grid.numel()
    
    @property
    def threshold(self) -> float:
        return min(self.base_threshold, self.mean)

    @property
    def device(self) -> torch.device:
        return self.grid.device

    @torch.no_grad()
    def update(self, sigma_fn: Callable[[torch.Tensor], torch.Tensor]):
        batch_shape = self.grid[0].size()
        for i in range(self.grid.size(0)):
            coords = -1. + 2. * (self.coords[i] + torch.rand_like(self.coords[i])) / self.size # jitter inside voxel 
            coords = coords.view(-1, 3).to(self.device)
            alpha = (1. - torch.exp(-sigma_fn(coords) * self.step_size)).view(batch_shape)
            self.grid[i] = torch.where(
                alpha > self.threshold,
                torch.ones_like(alpha, device=self.device),
                self.decay * self.grid[i]
            )
        self.mean = self.grid.mean().item()

    @torch.no_grad()
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """"coords: [..., 3], normalized [-1,1] coordinates"""
        new_shape = coords.shape[:-1]
        values = torch.nn.functional.grid_sample(
            self.grid[None,None,...],
            coords.view(1,-1,1,1,3),
            align_corners=True
        ).view(new_shape).contiguous()
        return values > self.threshold

@dataclass
class RayProvider():
    occupancy_grid: OccupancyGrid
    contraction: Contraction
    ray_marcher: RayMarcher

    @torch.no_grad()
    def __call__(self, rays_o: torch.Tensor, rays_d: torch.Tensor, training: bool):
        """Return packed_samples [n_samples, 7], where packed_samples[:,:3] are the samples
        origins, packed_samples[:,3:6] are samples directions, and packed_samples[:,6] are samples
        step sizes. packing_info is a [n_rays, 2] tensor where packing_info[:,0] is the starting
        position of samples form each ray, and packing_info[:,1] is the number of samples"""
        # Generate a grid [n_rays, n_samples, 3] of samples
        t_values, step_sizes = self.ray_marcher(rays_o, rays_d)
        if training: # jitter samples along ray when training
            t_values = t_values + torch.rand_like(t_values) * step_sizes
        samples = rays_o[:,None,:] + rays_d[:,None,:] * t_values[...,None]
        samples, marcher_mask = self.contraction(samples)
        mask = self.occupancy_grid(samples) if marcher_mask is None else marcher_mask & self.occupancy_grid(samples)

        # Pack the samples grid into 1D array, removing masked samples
        rays_count = torch.sum(mask, dim=-1, dtype=torch.int)
        rays_start = torch.cumsum(rays_count, dim=0, dtype=torch.int) - rays_count
        packing_info = torch.stack([rays_start, rays_count], -1)

        packed_d = torch.repeat_interleave(rays_d, rays_count, 0)
        packed_o = samples[mask]
        packed_steps = step_sizes[mask]
        packed_samples = torch.cat([packed_o, packed_d, packed_steps[:,None]], -1)

        return packed_samples, packing_info

"""NEURAL RENDERING LOGIC: Ray rendering, Ray weights computation"""

class NerfWeights(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, sigmas: torch.Tensor, steps: torch.Tensor, info: torch.Tensor, threshold: float) -> torch.Tensor: # type: ignore
        sigmas = sigmas.contiguous()
        steps = steps.contiguous()
        info = info.contiguous()
        weights = _cuda.compute_weights_fwd(sigmas, steps, info, threshold) # type: ignore
        ctx.save_for_backward(sigmas, steps, info, weights)
        return weights 

    @staticmethod
    def backward(ctx: Any, grad_weights: torch.Tensor): # type: ignore
        grad_weights = grad_weights.contiguous()
        sigmas, steps, info, weights = ctx.saved_tensors
        grad_sigmas = _cuda.compute_weights_bwd(sigmas, steps, info, weights, grad_weights) # type: ignore
        return grad_sigmas, None, None, None

class NerfRenderer(torch.nn.Module):
    def __init__(
        self,
        feature_module: torch.nn.Module,
        sigma_decoder: torch.nn.Module,
        rgb_decoder: torch.nn.Module,
        bg_color: torch.Tensor | None = None,
    ):
        super().__init__()
        self.feature_module = feature_module
        self.sigma_decoder = sigma_decoder
        self.rgb_decoder = rgb_decoder
        self.bg_color = bg_color

        assert hasattr(self.feature_module, "feature_dim"), "feature module requires a feature_dim attribute"

    def forward(
        self,
        packed_samples: torch.Tensor, # [n_samples, 7]
        packing_info: torch.Tensor, # [n_rays, 2]
        early_termination_threshold: float = 0., # TODO: add back 1e-4
    ) -> torch.Tensor:
        device = packed_samples.device
        n_samples = packed_samples.size(0)
        n_rays = packing_info.size(0)

        samples_features = self.feature_module(packed_samples[:,:3])
        samples_sigmas = self.sigma_decoder(samples_features).squeeze()

        weights: torch.Tensor = NerfWeights.apply(samples_sigmas, packed_samples[:,6], packing_info, early_termination_threshold) # type: ignore
        mask = weights > 0.

        samples_rgbs = torch.zeros((n_samples,3), device=device)
        if mask.any(): # compute rgb for remaining samples
            samples_rgbs[mask] = self.rgb_decoder(samples_features[mask], packed_samples[:,3:6][mask])
            samples_rgbs = samples_rgbs * weights[:,None]

        # TODO: cuda kernel this
        rendered_rgbs = torch.zeros((n_rays, 3), device=device)
        indices = torch.repeat_interleave(torch.arange(n_rays, device=device), packing_info[:,1])
        rendered_rgbs.index_add_(0, indices, samples_rgbs)

        if self.bg_color is not None:
            opacities = torch.zeros(n_rays, device=device)
            opacities.index_add_(0, indices, weights)
            # TODO: learn a background color ?
            rendered_rgbs = rendered_rgbs + self.bg_color.to(device) * (1 - opacities[:,None])

        return rendered_rgbs
