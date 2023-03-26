"""
Data handling formats, with implementations for Synthetic-NeRF, Mip-NeRF-360, and custom images.
All data are parsed into a common NerfData struct which is basically images+poses
"""

from __future__ import annotations

import numpy as np
import json
import torch
import gc
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Tuple, List, Optional
from pathlib import Path
from PIL import Image

@dataclass
class Intrinsics:
    fx: float; fy: float; cx: float; cy: float; w: int; h: int

    @property
    def matrix(self):
        return torch.tensor([
            [self.fx, 0., self.cx],
            [0., self.fy, self.cy],
            [0., 0., 1.]
        ])

@dataclass
class NerfData():
    """Nerf datas can either be labeled data (image_paths not None) when the
    ground truth colors are provided, or unlabeled data (image_paths None) when
    you want to generate the images for certain poses. Currently supports variable
    intrinsics images, will see if it's ever useful
    """
    cameras: torch.Tensor # [n_images, 4, 4] camera matrices
    intrinsics: Intrinsics | List[Intrinsics]
    imgs: Optional[List[torch.Tensor]] = None # [n_images], [h, w, 3] list of RGB HWC [0,1] images
    distortion: torch.Tensor | None = None
    camera_model: str = "OPENCV"
    bg_color: torch.Tensor | None = None # normalized [0,1] background rgb color

    def __post_init__(self):
        rays = [self.generate_rays(k) for k in range(len(self))]
        self.rays_o = [x[0] for x in rays]
        self.rays_d = [x[1] for x in rays]

    @property
    def shape(self) -> torch.Tensor:
        """Return list of shapes of img"""
        n = len(self)
        if isinstance(self.intrinsics, Intrinsics):
            shapes = torch.tensor([self.intrinsics.w, self.intrinsics.h]).expand(n, -1)
        else:
            shapes = torch.tensor([[K.w, K.h] for K in self.intrinsics])
        # one line: shapes = torch.tensor([[K.w, K.h] for K in (self.intrinsics if isinstance(self.intrinsics, List) else [self.intrinsics for _ in range(n)])])
        return shapes

    def idx_intrinsics(self, idx: int) -> Intrinsics:
        return self.intrinsics if isinstance(self.intrinsics, Intrinsics) else self.intrinsics[idx]

    def generate_rays(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        camera = self.cameras[idx]
        intrinsic = self.intrinsics[idx] if isinstance(self.intrinsics, List) else self.intrinsics
        # Generate ray directions
        center = torch.tensor([intrinsic.cx, intrinsic.cy], dtype=torch.float)
        focal = torch.tensor([intrinsic.fx, -intrinsic.fy], dtype=torch.float)
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(intrinsic.w, dtype=torch.float),
                torch.arange(intrinsic.h, dtype=torch.float),
                indexing="xy"
            ), -1
        )
        if self.distortion is not None:
            assert self.camera_model is not None, 'need a camera model'
            try:
                import cv2
            except ImportError as e:
                raise NotImplementedError('you need to install opencv for distortion correction')
            shape = grid.size()
            points = grid.view(-1, 2).numpy()
            K = intrinsic.matrix.numpy()
            distortion = self.distortion.numpy()
            if self.camera_model == "OPENCV":
                grid = torch.tensor(cv2.undistortPoints(points, K, distortion)).reshape(shape)
            elif self.camera_model == "OPENCV_FISHEYE":
                grid = torch.tensor(cv2.fisheye.undistortPoints(points, K, distortion)).reshape(shape)

        grid = (grid - center + 0.5) / focal
        grid = torch.nn.functional.pad(grid, (0,1), 'constant', -1.) # pad with 1 to get 3d coordinated

        # Apply camera transformation
        R, t = camera[:3,:3], camera[:3,3]
        d = grid @ R.T
        d /= torch.norm(d, dim=-1, keepdim=True) # normalize
        o = torch.broadcast_to(t, d.shape)
        return o, d
    
    def scene_scale(self) -> float:
        return torch.max(torch.var(self.cameras[:, :3, 3], 0)).item()
    
    def __len__(self):
        return len(self.cameras)
    
    def __getitem__(self, idx):
        o, d = self.generate_rays(idx)
        data = { "rays_o": o, "rays_d": d }
        if self.imgs is not None:
            data["rgbs"] = self.imgs[idx]
        return data

# Data from https://www.matthewtancik.com/nerf
def parse_nerf_synthetic(
    scene_path: Path,
    split: str = "train",
    bg_color: Tuple[int,int,int] = (255, 255, 255)
) -> NerfData:
    bg_color_tensor = torch.tensor(bg_color, dtype=torch.float) / 255.
    imgs, cameras = [], []
    intrinsics = None

    with open(scene_path / f"transforms_{split}.json") as f_in:
        data = json.load(f_in)

    for frame in data['frames']:
        image_path = (scene_path / frame['file_path']).with_suffix('.png')
        with Image.open(image_path) as img:
            if intrinsics is None:
                w, h = img.size
                camera_angle_x = data['camera_angle_x']
                focal = w / (2. * np.tan(0.5 * camera_angle_x))
                intrinsics = Intrinsics(focal, focal, w/2., h/2., w, h)

            if img.mode == "RGBA":
                bg = Image.new('RGBA', img.size, bg_color)
                img = Image.alpha_composite(bg, img).convert('RGB')
            torch_img = torch.from_numpy(np.array(img, dtype=np.single)) # TODO : copy?
            torch_img /= 255.
            imgs.append(torch_img)
        cameras.append(frame['transform_matrix'])

    assert intrinsics is not None
    return NerfData(
        imgs=imgs,
        cameras=torch.tensor(cameras, dtype=torch.float),
        intrinsics=intrinsics,
        bg_color=bg_color_tensor
    )

# nerfstudio data format, use ns-process-data to generate data from images/videos
# https://docs.nerf.studio/en/latest/quickstart/custom_dataset.html 
def parse_nerfstudio(
    scene_path: Path,
) -> NerfData:
    with open(scene_path / f"transforms.json") as f_in:
        data = json.load(f_in)

    intrinsics = Intrinsics(
        fx=data['fl_x'],
        fy=data['fl_y'],
        cx=data['cx'],
        cy=data['cy'],
        w=data['w'],
        h=data['h'],
    )

    camera_model = data.get("camera_model", "OPENCV")
    if camera_model == 'OPENCV':
        distortion = torch.tensor([data[coef] for coef in ['k1', 'k2', 'p1', 'p2']])
    elif camera_model == 'OPENCV_FISHEYE':
        distortion = torch.tensor([data[coef] for coef in ['k1', 'k2', 'k3', 'k4']])
    else:
        raise NotImplementedError(f'camera model {camera_model} not implemented')

    imgs, cameras = [], []
    for frame in data['frames']:
        image_path = scene_path / frame['file_path']
        with Image.open(image_path) as img:
            if img.mode == "RGBA":
                bg = Image.new('RGBA', img.size, (255,255,255))
                img = Image.alpha_composite(bg, img).convert('RGB')
            torch_img = torch.from_numpy(np.array(img, dtype=np.single)) # TODO : copy?
            torch_img /= 255.
            imgs.append(torch_img)
        cameras.append(frame['transform_matrix'])

    return NerfData(
        cameras=torch.tensor(cameras, dtype=torch.float),
        intrinsics=intrinsics,
        imgs=imgs,
        distortion=distortion,
        camera_model=camera_model
    )