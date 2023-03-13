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
"""

from typing import Callable, List
import torch
import math

# TODO : should occupancy grid accept normalized [0,1] coordinates 
# or should it handle contraction itself ?
class OccupancyGrid(torch.nn.Module):
    def __init__(self, size: List[int] | int):
        super().__init__()
        size = size if isinstance(size, List) else [size, size, size]
        self.n_voxels = math.prod(size)
    
        self.grid = torch.zeros(size, dtype=torch.float)
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

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """"coords: [n, 3], normalized [-1,1] coordinates"""
        values = torch.nn.functional.grid_sample(
            # self.grid[None,None,...]
            self.grid.unsqueeze(0).unsqueeze(0),
            coords.view(1,-1,1,1,3),
            mode="bilinear", align_corners=False # TODO: align_corners=True?
        ).view(-1)
        return values