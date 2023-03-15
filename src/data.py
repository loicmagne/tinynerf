"""
Data handling formats, with implementations for Synthetic-NeRF, Mip-NeRF-360, and custom images.
All data are parsed into a common NerfData struct which is basically images+poses
"""

from __future__ import annotations

import numpy as np
import json
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Tuple, List, Optional
from pathlib import Path
from PIL import Image

@dataclass
class Intrinsics:
    fx: float; fy: float; cx: float; cy: float; w: int; h: int

@dataclass
class NerfData:
    """Nerf datas can either be labeled data (image_paths not None) when the
    ground truth colors are provided, or unlabeled data (image_paths None) when
    you want to generate the images for certain poses. Currently supports variable
    intrinsics images, will see if it's ever useful
    """
    
    cameras: List[torch.Tensor] # [n_images, 4, 4] camera matrices
    intrinsics: Intrinsics | List[Intrinsics]
    imgs: Optional[List[torch.Tensor]] = None # [n_images], [h, w, 3] list of RGB HWC [0,1] images
    
    @property
    def n_img(self):
        return len(self.cameras)

    @property
    def shape(self) -> torch.Tensor:
        """Return list of shapes of img"""
        n = self.n_img
        if isinstance(self.intrinsics, Intrinsics):
            shapes = torch.tensor([self.intrinsics.w, self.intrinsics.h]).expand(n, -1)
        else:
            shapes = torch.tensor([[K.w, K.h] for K in self.intrinsics])
        # one line: shapes = torch.tensor([[K.w, K.h] for K in (self.intrinsics if isinstance(self.intrinsics, List) else [self.intrinsics for _ in range(n)])])
        return shapes

    def generate_rays(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        rays_o, rays_d = [], []
        for i, camera in enumerate(self.cameras):
            intrinsic = self.intrinsics[i] if isinstance(self.intrinsics, List) else self.intrinsics
            # Generate ray directions
            center = torch.tensor([intrinsic.cx, intrinsic.cy], dtype=torch.float)
            focal = torch.tensor([intrinsic.fx, intrinsic.fy], dtype=torch.float)
            # TODO : indexing might be wrong
            grid = torch.stack(torch.meshgrid(torch.arange(intrinsic.w, dtype=torch.float), torch.arange(intrinsic.h, dtype=torch.float), indexing="xy"), -1)
            grid = (grid - center) / focal
            grid = torch.nn.functional.pad(grid, (0,1), 'constant', 1.) # pad with 1 to get 3d coordinated

            # Apply camera transformation
            d = (grid @ camera[:3,:3].T)
            d = d / torch.norm(d, dim=-1, keepdim=True) # normalize
            o = torch.zeros_like(d) + camera[:3, 3]

            rays_o.append(d)
            rays_d.append(o)
        return rays_o, rays_d

class ImagesDataset(Dataset):
    def __init__(self, data: NerfData):
        # Note: h,w can be different for each image
        self.rays_o, self.rays_d = data.generate_rays() # [n_images][h, w, 3]
        self.rgbs = data.imgs # [n_images][h, w, 3], None when doing novel view synthesis

    def __len__(self):
        return len(self.rays_o)

    def __getitem__(self, idx):
        data = {
            "rays_o": self.rays_o[idx],
            "rays_d": self.rays_d[idx],
        }
        if self.rgbs is not None:
            data["rgbs"] = self.rgbs[idx]
        return data

class RaysDataset(Dataset):
    def __init__(self, data: NerfData):
        rays_o, rays_d = data.generate_rays()
        self.rays_o = torch.cat([t.view(-1, 3) for t in rays_o]) # [n_rays, 3]
        self.rays_d = torch.cat([t.view(-1, 3) for t in rays_d]) # [n_rays, 3]
        self.rgbs = torch.cat([t.view(-1, 3) for t in data.imgs]) if data.imgs is not None else None # [n_rays, 3]

    def __len__(self):
        return self.rays_o.size(0)

    def __getitem__(self, idx):
        data = {
            "rays_o": self.rays_o[idx],
            "rays_d": self.rays_d[idx],
        }
        if self.rgbs is not None:
            data["rgbs"] = self.rgbs[idx]
        return data

def parse_nerf_synthetic(scene_path: Path, split: str = "train") -> NerfData:
    with open(scene_path / f"transforms_{split}.json") as f_in:
        data = json.load(f_in)
    imgs, cameras = [], []
    intrinsics = None
    for frame in data['frames']:
        image_path = (scene_path / frame['file_path']).with_suffix('.png')
        with Image.open(image_path) as img:
            if intrinsics is None:
                w, h = img.size
                camera_angle_x = data['camera_angle_x']
                focal = w / (2. * np.tan(0.5 * camera_angle_x))
                intrinsics = Intrinsics(focal, focal, w/2., h/2., w, h)

            if img.mode == "RGBA":
                bg = Image.new('RGBA', img.size, (255, 255, 255))
                img = Image.alpha_composite(bg, img).convert('RGB')
            torch_img = torch.from_numpy(np.array(img, dtype=np.single)) # TODO : copy?
            torch_img /= 255.
            imgs.append(torch_img)
        cameras.append(torch.tensor(frame['transform_matrix']))
    assert intrinsics is not None
    return NerfData(imgs=imgs, cameras=cameras, intrinsics=intrinsics)
