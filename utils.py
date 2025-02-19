import numpy as np
import pandas as pd

import torch

from sklearn.preprocessing import StandardScaler

def load_data():
    df = pd.read_csv("./data/boston_house_prices.csv", skiprows=1, dtype=np.float32)
    df.rename(columns={"MEDV" : "TARGET"}, inplace=True)

    # Standardization - mean = 0, variance = 1
    scaler = StandardScaler()
    scaler.fit(df.values[:, :-1])
    df.values[:, :-1] = scaler.transform(df.values[:, :-1])

    data = torch.from_numpy(df.values).float()

    x, y = data[:, :-1], data[:, -1:]

    return x, y

