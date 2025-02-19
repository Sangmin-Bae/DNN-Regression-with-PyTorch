import argparse

import torch

from utils import load_data

def argument_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--model_fn", required=True, help="model_file_name")
    p.add_argument("--gpu_id", type=int, default=0 if torch.cuda.is_available() else -1)

    config = p.parse_args()

    return config

def main(config):
    # Set device
    device = torch.device("cpu") if config.gpu_id < 0  else torch.device(f"cuda:{config.gpu_id}")
    print(f"Device : {device}")

    # Load Data
    x, y = load_data()

    print(f"Train Data : {x.shape}")
    print(f"Target Data : {y.shape}")

    # Define model

    # Set optimizer, criterion

    # Train model

    # Save best model

if __name__ == "__main__":
    config = argument_parser()
    main(config)