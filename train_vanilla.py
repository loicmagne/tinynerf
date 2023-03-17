#python train_vanilla.py --data data/lego --output data/output --method vanilla --steps 10000 --batch_size 4096 --eval_every 250 --eval_n 5 --occupancy_res 128
import argparse
from pathlib import Path
from src.train import train_vanilla, VanillaTrainConfig
from src.data import parse_nerf_synthetic, RaysDataset, ImagesDataset

def get_config() -> VanillaTrainConfig:
    parser = argparse.ArgumentParser(prog='tinynerf', description='Train nerf')
    
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--eval_every', type=int)
    parser.add_argument('--eval_n', type=int)
    parser.add_argument('--occupancy_res', type=int)

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
        args.eval_every,
        args.eval_n,
        args.occupancy_res
    )

if __name__ == '__main__':
    train_vanilla(get_config())