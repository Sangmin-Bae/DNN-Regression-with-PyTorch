import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import MyDNNModel
from trainer import Trainer

from utils import load_boston_house_prices_data

def argument_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--model_fn", required=True, help="model_file_name")
    p.add_argument("--data_number", type=int, default=1, help="data number [1: boston / 2: california]")
    p.add_argument("--gpu_id", type=int, default=0 if torch.cuda.is_available() else -1, help="number_of_gpu_id")
    p.add_argument("--batch_size", type=int, default=256, help="mini_batch_size")
    p.add_argument("--n_epochs", type=int, default=100000, help="number_of_epochs")
    p.add_argument("--lr", type=float, default=1e-4, help="learning_rate")
    p.add_argument("--print_interval", type=int, default=5000, help="number_of_print_interval")

    config = p.parse_args()

    return config

def main(config):
    # Set device
    device = torch.device("cpu") if config.gpu_id < 0  else torch.device(f"cuda:{config.gpu_id}")
    print(f"Device : {device}")

    # Load Data
    x, y = load_boston_house_prices_data()

    print(f"Train Data : {x.shape}")
    print(f"Target Data : {y.shape}")

    # Define model
    model = MyDNNModel(input_size=x.size(-1), output_size=y.size(-1)).to(device)
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
    config = argument_parser()
    main(config)