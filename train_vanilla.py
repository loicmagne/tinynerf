#python train_vanilla.py --data data/lego --output data/output --method vanilla --batch_size 4096 --scene_type aabb
import argparse
from pathlib import Path
from src.train import train_vanilla, VanillaTrainConfig
from src.data import parse_nerf_synthetic, RaysDataset, ImagesDataset

def get_config() -> VanillaTrainConfig:
    parser = argparse.ArgumentParser(prog='tinynerf', description='Train nerf')
    
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--n_samples', type=int, default=400)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--eval_n', type=int, default=1)
    parser.add_argument('--occupancy_res', type=int, default=128)
    parser.add_argument('--scene_type', type=str)

    args = parser.parse_args()

    train_rays = RaysDataset(parse_nerf_synthetic(Path(args.data), 'train'))
    train_images = ImagesDataset(parse_nerf_synthetic(Path(args.data), 'train'))
    test_images = ImagesDataset(parse_nerf_synthetic(Path(args.data), 'test'))

    return VanillaTrainConfig(
        train_rays,
        train_images,
        test_images,
        Path(args.output),
        args.method,
        args.steps,
        args.batch_size,
        args.n_samples,
        args.eval_every,
        args.eval_n,
        args.occupancy_res,
        args.scene_type
    )

if __name__ == '__main__':
    import torch
    import numpy as np
    import random
    import os

    SEED = int(os.environ.get('SEED', 0))
    if SEED != 0:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

    train_vanilla(get_config())