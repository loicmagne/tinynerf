""" TODO: remove
si on décompose l'implémentation en étapes on a:
    - interface pour les datasets qui permet d'accéder aux samples/images/positions
    - boucle d'entrainement classique du ML train/test en itérant sur les rayons et en appelant une fonction render
    - la logique de rendering NeRF, raymarching + accumulation
    - contractions    
    -> partie encore assez vague pour moi, la logique de space skipping / proposal sampling qui à l'air fondamental pour avoir un truc efficace, mais assez chiant à implémenter

    - implémentation de la méthode en elle même tensorf kplanes etc.. donc simplement un moyen de passer des coordonnées à couleur/densité

TODO:
    - datasets loading / common interface (synthetic, mip nerf, etc.)
    - typical DL training loop
    - complete metric computation
    - rendering logic: raymarching + accumulation
    - density grid for space skipping, prposal sampling ?
    - contractions
    - kplanes actual implementation


- comprendre le space skipping de tensorf/kplanes


pour le space skipping, il y'a 2 méthodes:
    - occupancy grid binaire qui stocke pour chaque voxel si un booléen qui vaut true si cet endroit en occupé. L'occupancy grid est ensuite utilisée dans le ray marching pour skip les voxels inutiles
    - proposal network, un NN est utilisé pour estimer la densité sur un rayon, qui est ensuite utilisé itérativement pour raffiner l'estimation de la densité. À la fin on sample la densité sur le rayon pour obtenir les samples  

la méthode proposal network a l'air plus simple à implémenter, au prix surement des performances, puisque ça rajoute un NN à évaluer. la méthode occupancy grid est surement plus chiante à implémenter, mais je dirais qu'elle est plus rapide + peut être accélérée sur gpu. la méthode proposal network alterne entre sampling et évaluation de NN donc je sais pas trop comment ça s'accélère avec CUDA

pour les scenes contractions, globalement c'est simple, on travaille toujours avec des rayons dans les coordonnées du monde normal, et quand on a besoin de contracter (parce qu'on veut query une grid ou autre) on contracte.

- comprendre le loading de datasets
- checker le raymarching de jaxnerf





we start with a bunch of rays of shape [n,3] for which we want to compute the color (i.e. sample along ray + accumulate)

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
    return torch.where(norm <= 1., coords, (2. - 1./norm) * coords / norm)

# TODO : should occupancy grid accept normalized [0,1] coordinates 
# or should it handle contraction itself ?
class OccupancyGrid(torch.nn.Module):
    def __init__(self, size: List[int] | int):
        super().__init__()
        size = size if isinstance(size, List) else [size, size, size]
        self.n_voxels = math.prod(size)
    
        self.grid = torch.ones(size, dtype=torch.float)
        self.size = torch.tensor(self.grid.size())
        self.stride = torch.tensor(self.grid.stride())
        self.coords = torch.stack(torch.meshgrid([
            torch.arange(size[0], dtype=torch.float),
            torch.arange(size[1], dtype=torch.float),
            torch.arange(size[2], dtype=torch.float)
        ], indexing="ij"), -1).view(-1, 3)

    # TODO : delete?
    def indices_to_coordinates(self, indices: torch.Tensor) -> torch.Tensor :
        """Turn indices (shape [n]) in range [0, n_voxels] to 3D coordinates (shape [n,3]) in the grid"""
        return torch.remainder(indices.unsqueeze(1) - self.stride, self.size)

    @torch.no_grad()
    def update(self, occupancy_fn: Callable[[torch.Tensor], torch.Tensor], threshold: float):
        coords = (self.coords + 0.5 + torch.randn_like(self.coords)) / self.size # jitter to sample different points, TODO: change to uniform
        self.grid = (occupancy_fn(coords) > threshold).view(self.grid.shape).float() # TODO: batch evaluation

    @torch.no_grad()
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """"coords: [n, 3], normalized [-1,1] coordinates"""
        values = torch.nn.functional.grid_sample(
            # self.grid[None,None,...]
            self.grid.unsqueeze(0).unsqueeze(0),
            coords.view(1,-1,1,1,3),
            mode="bilinear", align_corners=False # TODO: align_corners=True?
        ).view(-1)
        return values


def clamped_exponential_stepping(delta_min: float, delta_max: float, t_0: float, exp: float):
    """Exponential stepping function as described in Instant-NGP paper, which
    is a regular exponential function but extended to a linear function after the derivative 
    is greater than delta_max or before it is lower than delta_min"""
    min_threshold = delta_min / np.log(t_0 * np.log(exp) * exp)
    max_threshold = delta_max / np.log(t_0 * np.log(exp) * exp)
    
    alpha_min = delta_min
    alpha_max = delta_max
    beta_min = np.power(exp, min_threshold) * t_0 - alpha_min * min_threshold
    beta_max = np.power(exp, max_threshold) * t_0 - alpha_max * max_threshold

    def f(x: torch.Tensor):
        return torch.where(
            x < min_threshold,
            alpha_min * x + beta_min,
            torch.where(
                x > max_threshold,
                alpha_max * x + beta_max,
                torch.exp(x * np.log(exp)) * t_0
            )
        )
    return f

class NerfRenderer(torch.nn.Module):
    def __init__(
        self,
        occupancy_grid: OccupancyGrid,
        feature_module: torch.nn.Module,
        sigma_decoder: torch.nn.Module,
        rgb_decoder: torch.nn.Module,
        contraction: Callable[[torch.Tensor], torch.Tensor] = mip360_contract
    ):
        super().__init__()
        self.occupancy_grid = occupancy_grid
        self.feature_module = feature_module
        self.sigma_decoder = sigma_decoder
        self.rgb_decoder = rgb_decoder
        self.contraction = contraction

    def forward(
        self,
        rays_o: torch.Tensor, # [n, 3]
        rays_d: torch.Tensor, # [n, 3]
        n_samples: int, # number of samples along each ray
        near: float, # minimum distance along ray to sample
        far: float, # maximum distance along ray to sample
        early_termination: float = 0.01 # early ray termination threshold
    ) -> torch.Tensor:
        device = rays_o.device
        n = rays_o.shape[0]

        # Generate samples along each ray
        # TODO : precompute and store t_values
        t_close = torch.linspace(near, near+1, n_samples // 2, dtype=torch.float, device=device)
        t_far  = torch.exp(torch.arange(n_samples // 2, dtype=torch.float, device=device) * np.log(1.+ 1./256.)) * (near + 1.)
        t_values = torch.cat([t_close, t_far])
        # TODO: distances are wrong for now since they don't account for ray direction norm -> should normalize rays_d?
        distances = t_values[1:] - t_values[:-1]
        t_values = t_values[:-1]
        n_samples -= 1
        
        # jitter samples along ray when training
        if self.training:
            t_values = t_values + torch.rand_like(t_values) * distances

        samples_grid = rays_o[:, None, :] + rays_d[:, None, :] * t_values[None, :, None]
        samples_grid = self.contraction(samples_grid)
        mask : torch.Tensor = self.occupancy_grid(samples_grid).view(n, n_samples).bool()

        assert hasattr(self.feature_module, "feature_dim"), "feature module requires a feature_dim attribute"
        samples_features = torch.zeros(n, n_samples, cast(int, self.feature_module.feature_dim), device=device)
        samples_sigmas = torch.zeros(n, n_samples, device=device)
        samples_rgbs = torch.zeros(n, n_samples, 3, device=device)

        # compute features and density for remaining samples
        samples_features[mask] = self.feature_module(samples_grid[mask])
        samples_sigmas[mask] = self.sigma_decoder(samples_features[mask]).squeeze()
        
        # compute transmittance and alpha
        alpha = -samples_sigmas * distances[None, :] # not actually alpha
        transmittance = torch.exp(torch.cumsum(alpha, 1))
        transmittance[:, 1:] = transmittance[:, :-1] # shift transmittance to the right
        transmittance[:, 0] = 1.
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
