import json
import argparse
from trainer import Trainer

def main(args):
    with open(args.config, 'r') as file:
        config = json.load(file)
    trainer = Trainer(depth=args.depth, config=config)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script for the model.')
    parser.add_argument('--config', type=str, default='configs/default_config.json', help='Path to the configuration file')
    parser.add_argument('--depth', type=int, default=4, choices=[1, 2, 3, 4, 5], help='Depth of the model')
    args = parser.parse_args()
    main(args)

