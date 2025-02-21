import torch
import torch.nn as nn

class MyDNNModel(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        super().__init__()

        self.linear1 = nn.Linear(input_size, 3)
        self.linear2 = nn.Linear(3, 3)
        self.linear3 = nn.Linear(3, 3)
        self.linear4 = nn.Linear(3, 3)
        self.linear5 = nn.Linear(3, 3)
        self.linear6 = nn.Linear(3, output_size)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        # |x| = (N, input_size)
        h = self.act(self.linear1(x))  # |h| = (N, 3)
        h = self.act(self.linear2(h))
        h = self.act(self.linear3(h))
        h = self.act(self.linear4(h))
        h = self.act(self.linear5(h))
        y = self.linear6(h)
        # |y = (N, output_size)

        return y

class MyDNNModelV2(nn.Module):
    def __init__(self, intput_size, output_size):
        self.input_size = intput_size
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