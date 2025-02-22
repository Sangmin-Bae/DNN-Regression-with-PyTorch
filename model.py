import torch
import torch.nn as nn

class MyDNNModel(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 6),
            nn.LeakyReLU(),
            nn.Linear(6, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 3),
            nn.LeakyReLU(),
            nn.Linear(3, self.output_size)
        )

    def forward(self, x):
        # |x| = (batch_size, input_size)
        y = self.layers(x)
        # |y| = (batch_size, output_size)

        return y