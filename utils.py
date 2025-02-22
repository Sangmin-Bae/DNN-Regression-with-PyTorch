import pandas as pd

import torch

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

def load_data():
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