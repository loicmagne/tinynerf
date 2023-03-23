#python train_vanilla.py --data data/lego --output data/output --method kplanes --batch_size 1024 --n_samples 1024 --scene_type aabb
import argparse
import uuid
from pathlib import Path
from src.train import train_vanilla, VanillaTrainConfig
from src.data import parse_nerf_synthetic, RaysDataset, ImagesDataset

def get_config() -> VanillaTrainConfig:
    parser = argparse.ArgumentParser(prog='tinynerf', description='Train nerf')
    
    parser.add_argument('--data', type=str, required=True, help='path to the data folder')
    parser.add_argument('--output', type=str, required=True, help='path to the output folder')
    parser.add_argument('--method', type=str, required=True, choices=['vanilla', 'kplanes', 'cobafa'])
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--n_samples', type=int, default=400, help='number of samples per ray')
    parser.add_argument('--eval_every', type=int, default=None, help='number of train steps between evaluations')
    parser.add_argument('--eval_n', type=int, default=1, help='number of images to evaluate on')
    parser.add_argument('--scene_type', type=str, default='aabb', choices=['aabb', 'unbounded'])

    args = parser.parse_args()

    train_rays = RaysDataset(parse_nerf_synthetic(Path(args.data), 'train'))
    test_images = ImagesDataset(parse_nerf_synthetic(Path(args.data), 'test'))

    return VanillaTrainConfig(
        train_rays=train_rays,
        test_images=test_images,
        output=Path(args.output),
        method=args.method,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        eval_every=args.eval_every,
        eval_n=args.eval_n,
        scene_type=args.scene_type
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

    config = get_config()

    while True:
        id = str(uuid.uuid4())[:8]
        experiment_name = f'{id}_{config.method}_{config.scene_type}_{config.n_samples}'
        if not (config.output / experiment_name).is_dir():
            break
    config.output = config.output / experiment_name
    config.output.mkdir(parents=True)

    print(f'Experiment saved to {config.output}')

    train_vanilla(config)