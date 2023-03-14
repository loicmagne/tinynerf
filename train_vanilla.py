# python train_vanilla.py --train tests/dummy/hotdog --test tests/dummy/hotdog --epochs 1 --batch_size 256 --eval_every 1 
import argparse
from pathlib import Path
from src.train import train_vanilla, VanillaTrainConfig
from src.data import parse_nerf_synthetic, NerfDataset

def get_config() -> VanillaTrainConfig:
    parser = argparse.ArgumentParser(prog='tinynerf', description='Train nerf')
    
    parser.add_argument('--steps', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--eval_every', type=int)
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--test', type=str, required=True)

    args = parser.parse_args()

    train_dataset = NerfDataset(parse_nerf_synthetic(Path(args.train), 'train'))
    test_dataset = NerfDataset(parse_nerf_synthetic(Path(args.test), 'train'))

    return VanillaTrainConfig(
        train_dataset,
        test_dataset,
        args.steps,
        args.batch_size,
        args.eval_every,
    )

if __name__ == '__main__':
    train_vanilla(get_config())