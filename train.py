import argparse

import torch

def argument_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--model_fn", required=True, help="model_file_name")
    p.add_argument("--gpu_id", type=int, default=0 if torch.cuda.is_available() else -1)

    config = p.parse_args()

    return config

def main(config):
    # Define device
    device = torch.device("cpu") if config.gpu_id < 0  else torch.device(f"cuda:{config.gpu_id}")

    # Load Data

    # Define model

    # Set optimizer, criterion

    # Train model

    # Save best model

if __name__ == "__main__":
    config = argument_parser()
    main(config)