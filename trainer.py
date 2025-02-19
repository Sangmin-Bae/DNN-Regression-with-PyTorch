import numpy as np

from copy import deepcopy

import torch
import torch.nn.functional as func

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

        super().__init__()

    def train(self, x, y, config):
        lowest_loss = np.inf
        best_model = None

        for idx in range(config.n_epochs):
            y_hat = self.model(x)
            loss = func.mse_loss(y_hat, y)

            # Initialize gradient
            self.optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # Gradient descent
            self.optimizer.step()

            if loss < lowest_loss:
                lowest_loss = loss
                best_model = deepcopy(self.model.state_dict())

            if (idx + 1) % config.print_interval == 0:
                print(f"Epoch {idx + 1} : loss={float(loss):.4e}")

        # Restore best model
        self.model.load_state_dict(best_model)