"""
Data handling formats, with implementations for Synthetic-NeRF, Mip-NeRF-360, and custom images.
All data are parsed into a common NerfData struct which is basically images+poses

--- idea buffer, TODO : REMOVE

- There should be a common dataset format which takes as input a list of images with their poses and gives a standard nerf dataset API, i.e. which can provide rays, colors, images etc.
- The issue is what should this Dataset class provide: the simple idea would be to only provide rendered rays of the form (origin, direction, pixel color) however doing so we lose the information about the original images, but we need those original images to compute some metrics, so it would be great to have a way to get back to the original images


datasets:
    - load every images and convert them to rays
    - store 3 tensors : rays, rgbs, image_idx, + the list of cameras
    - when sampling rays we just return rays and rgbs
    - when sampling images 

"""
from __future__ import annotations

import numpy as np
import json
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Optional, cast
from pathlib import Path
from PIL import Image

class NerfDataset(Dataset):
    def __init__(self, data: NerfData):
        self.rays_o: torch.Tensor # [n_rays, 3]
        self.rays_d: torch.Tensor # [n_rays, 3]
        self.rgbs: torch.Tensor | None # [n_rays, 3]
        self.indices: torch.Tensor # [n_rays, 1] index of the image the ray belongs to
        self.cameras: List[torch.Tensor] # [n_images, 4, 4] camera matrices
        self.shape: torch.Tensor

        self.load_and_transform(data)

    def load_and_transform(self, data: NerfData):
        self.rays_o, self.rays_d = data.generate_rays()
        self.indices = torch.repeat_interleave(torch.arange(data.n_img), torch.prod(data.shape,-1)).short()
        self.shape = data.shape

        if data.img_paths is not None:
            imgs = []
            for path in data.img_paths:
                with Image.open(path) as img:
                    torch_img = torch.from_numpy(np.array(img)) # TODO : copy? dtype=?
                    imgs.append(torch_img)
            self.rgbs = torch.cat([t.flatten() for t in imgs])

    def recover_images(self, rgbs: torch.Tensor, indices: torch.Tensor | None = None) -> List[torch.Tensor]:
        """ Reshape a list of (rgb, camera index) into a list of images """
        if indices is None:
            indices = self.indices
        
        img_indices : torch.Tensor = torch.unique(indices)
        imgs = []
        for idx in img_indices:
            imgs.append(rgbs[indices==idx].view(self.shape[idx].tolist()))
        return imgs

    def __len__(self):
        return self.indices.size(0)

    def __getitem__(self, idx):
        data = {
            "rays_o": self.rays_o[idx],
            "rays_d": self.rays_d[idx],
            "indices": self.indices[idx]
        }
        if self.rgbs is not None:
            data = self.rgbs[idx],
        return data


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
    
    cameras: List[torch.Tensor]
    intrinsics: Intrinsics | List[Intrinsics]
    img_paths: Optional[List[Path]] = None
    
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
        # could replace with one liner
        # shapes = torch.tensor([[K.w, K.h] for K in self.intrinsics if isinstance(self.intrinsics, List) else [self.intrinsics for _ in range(n)]])
        return shapes

    def generate_rays(self):
        rays_d, rays_o = [], []
        for i, c in enumerate(self.cameras):
            intrinsic = self.intrinsics[i] if isinstance(self.intrinsics, List) else self.intrinsics
            camera = self.cameras[i]
            # Generate ray directions
            center = torch.tensor([intrinsic.cx, intrinsic.cy])
            focal = torch.tensor([intrinsic.fx, intrinsic.fy])
            grid = torch.stack(torch.meshgrid(torch.arange(intrinsic.w), torch.arange(intrinsic.h)), -1)
            grid = (grid - center) / focal
            grid = torch.nn.functional.pad(grid, (0,1), 'constant', 1.) # pad with 1 to get 3d coordinated

            # Apply camera transformation
            d = (grid @ camera[:3,:3].T).view(-1,3)
            o = torch.zeros_like(d) + camera[:3, 3]

            # TODO: set rays origin to the near plane instead of camera origin
            
            rays_d.append(d)
            rays_o.append(o)
        rays_d, rays_o = torch.cat(rays_d), torch.cat(rays_o)
        return rays_d, rays_o

def parse_nerf_synthetic(scene_path: Path, split: str = "train") -> NerfData:
    with open(scene_path / f"transforms_{split}.json") as f_in:
        data = json.load(f_in)
    image_paths, cameras = [], []
    for frame in data['frames']:
        image_paths.append(scene_path / frame['file_path'])
        cameras.append(torch.tensor(frame['transform_matrix']))
    # Compute intrinsics
    with Image.open(image_paths[0]) as img:
        w, h = img.size
        camera_angle_x = data['camera_angle_x']
        focal = w / (2. * np.tan(0.5 * camera_angle_x))
        intrinsics = Intrinsics(focal, focal, w/2., h/2., w, h)
    return NerfData(img_paths=image_paths, cameras=cameras, intrinsics=intrinsics)
