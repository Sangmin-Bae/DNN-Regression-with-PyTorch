import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim

from model import MyDNNModelV2
from trainer import Trainer

from utils import load_data

def argument_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--config", type=str, required=True, help="config_file_path")

    args = p.parse_args()

    return args

def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config

def main(config):
    # Set device
    device = torch.device("cpu") if config.gpu_id < 0  else torch.device(f"cuda:{config.gpu_id}")
    print(f"Device : {device}")

    # Load Data
    x, y = load_data(config.data_number)

    print(f"Train Data : {x.shape}")
    print(f"Target Data : {y.shape}")

    # Define model
    model = MyDNNModelV2(input_size=x.size(-1), output_size=y.size(-1)).to(device)
    print(f"Model: {model}")

    # Set optimizer
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    print(f"Optimizer: {optimizer}")

    # Train model
    trainer = Trainer(model, optimizer)
    trainer.train(x.to(device), y.to(device), config)

    # Save best model
    torch.save({
        "model": trainer.model.state_dict(),
        "opt": optimizer.state_dict(),
        "config": config
    }, config.model_fn)

if __name__ == "__main__":
    args = argument_parser()
    config = load_config(args.config)
    main(config)