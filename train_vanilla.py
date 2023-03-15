# python train_vanilla.py --train tests/dummy/hotdog --test tests/dummy/hotdog --output data/output --method vanilla --steps 1000 --batch_size 64 --eval_every 300 --occupancy_res 32 
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

    train_dataset = RaysDataset(parse_nerf_synthetic(Path(args.data), 'train'))
    test_dataset = ImagesDataset(parse_nerf_synthetic(Path(args.data), 'train'))

    return VanillaTrainConfig(
        train_dataset,
        test_dataset,
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