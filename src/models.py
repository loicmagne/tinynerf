import torch
from typing import List
from dataclasses import dataclass

"""Vanilla NeRF"""

class PositionalEncoding(torch.nn.Module):
    def __init__(self, freqs: torch.Tensor):
        super().__init__()
        self.freqs = freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[...,None] * self.freqs
        x = torch.cat([torch.sin(x), torch.cos(x)], -1)
        return x.flatten(-2)
    
class VanillaFeatureMLP(torch.nn.Module):
    def __init__(self, freqs: torch.Tensor, hidden_features: List[int]):
        super().__init__()
        in_features = freqs.size(0) * 2 * 3
        self.net = torch.nn.Sequential(
            PositionalEncoding(freqs),
            torch.nn.Linear(in_features, hidden_features[0]),
            *[torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_features[k], hidden_features[k+1]),
            ) for k in range(len(hidden_features)-1)],
        )
        self.feature_dim = hidden_features[-1]

    def forward(self, x):
        return self.net(x)
    
class VanillaOpacityDecoder(torch.nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, 1),
            torch.nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.net(x)
    
class VanillaColorDecoder(torch.nn.Module):
    def __init__(self, freqs: torch.Tensor, in_features: int, hidden_features: List[int]):
        super().__init__()
        self.pe = PositionalEncoding(freqs)
        self.total_features = in_features + freqs.size(0) * 2 * 3
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.total_features, hidden_features[0]),
            *[torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_features[k], hidden_features[k+1]),
            ) for k in range(len(hidden_features)-1)],
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features[-1], 3),
            torch.nn.Sigmoid(),
        )
    
    def forward(self, features: torch.Tensor, rays_d: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.pe(rays_d), features], -1)
        return self.net(x)