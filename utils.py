import numpy as np
import pandas as pd

import torch

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

def load_boston_house_prices_data():
    df = pd.read_csv("./data/boston_house_prices.csv", skiprows=1, dtype=np.float32)
    df.rename(columns={"MEDV" : "TARGET"}, inplace=True)

    # Standardization - mean = 0, variance = 1
    scaler = StandardScaler()
    scaler.fit(df.values[:, :-1])
    df.values[:, :-1] = scaler.transform(df.values[:, :-1])

    data = torch.from_numpy(df.values).float()

    x, y = data[:, :-1], data[:, -1:]

    return x, y

def load_california_housing_data():
    california = fetch_california_housing()

    df = pd.DataFrame(california.data, columns=california.feature_names)
    df.tail()

    # Standardization - mean = 0, variance = 1
    scaler = StandardScaler()
    scaler.fit(df.values[:, :-1])
    df.values[:, :-1] = scaler.transform(df.values[:, :-1])

    # Add target column
    df["Target"] = california.target

    # Dataframe to Tensor
    data = torch.from_numpy(df.values).float()

    x, y = data[:, :-1], data[:, -1:]

    return x, y

def load_data(data_number):
    if data_number == 1:
        x, y = load_boston_house_prices_data()
    elif data_number == 2:
        x, y = load_california_housing_data()
    else:
        raise ValueError()

    return x, y