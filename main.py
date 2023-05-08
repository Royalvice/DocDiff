from src.config import load_config
from src.train import train, test
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='conf.yml', help='path to the config.yaml file')
    args = parser.parse_args()
    config = load_config(args.config)
    print('Config loaded')
    mode = config.MODE
    if mode == 1:
        train(config)
    else:
        test(config)
if __name__ == "__main__":
    main()