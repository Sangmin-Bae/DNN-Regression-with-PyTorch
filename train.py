import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim

from model import MyDNNModel
from trainer import Trainer

from utils import load_data
from utils import split_data

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
    device = torch.device("cpu") if config["gpu_id"] < 0  else torch.device(f"cuda:{config['gpu_id']}")
    print(f"Device : {device}")

    # Load Data
    x, y = load_data()

    # Split data
    x, y = split_data(x, y, device, config["train_ratio"])
    train_x, valid_x = x[0], x[1]
    train_y, valid_y = y[0], y[1]

    # Define model
    model = MyDNNModel(input_size=train_x.size(-1), output_size=train_y.size(-1)).to(device)
    print(f"Model: {model}")

    # Set optimizer
    optimizer = optim.Adam(model.parameters())
    print(f"Optimizer: {optimizer}")

    # Train model
    trainer = Trainer(model, optimizer)
    trainer.train(
        train_data=(train_x, train_y),
        valid_data=(valid_x, valid_y),
        config=config
    )

    # Save best model
    torch.save({
        "model": trainer.model.state_dict(),
        "opt": optimizer.state_dict(),
        "config": config
    }, config["model_fn"])

if __name__ == "__main__":
    args = argument_parser()
    config = load_config(args.config)
    main(config)