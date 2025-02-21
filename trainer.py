import numpy as np

from copy import deepcopy

import torch
import torch.nn.functional as func

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

        super().__init__()

    @ staticmethod
    def _batchify(self, x, y, batch_size, random_split=True):
        # Shuffle the index to feed-forward
        if random_split:
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices)
            y = torch.index_select(y, dim=0, index=indices)

        # |x| = (batch_size, input_size)
        x = x.split(batch_size, dim=0)
        # |y| = (batch_size, output_size)
        y = y.split(batch_size, dim=0)

        return x,y

    def train(self, x, y, config):
        lowest_loss = np.inf
        best_model = None

        for idx in range(config.n_epochs):
            x, y = self._batchify(x, y, config.batch_size)

            total_loss = 0

            for x_i, y_i in zip(x, y):
                y_hat_i = self.model(x_i)
                loss = func.mse_loss(y_hat_i, y_i)

                # Initialize gradient
                self.optimizer.zero_grad()

                # Backpropagation
                loss.backward()

                # Gradient descent
                self.optimizer.step()

                total_loss += float(loss)  # prevent memory leak by gradient

            loss = total_loss / len(x)  # mean loss

            if loss < lowest_loss:
                lowest_loss = loss
                best_model = deepcopy(self.model.state_dict())

            if (idx + 1) % config.print_interval == 0:
                print(f"Epoch {idx + 1} : loss={float(loss):.4e}")

        # Restore best model
        self.model.load_state_dict(best_model)