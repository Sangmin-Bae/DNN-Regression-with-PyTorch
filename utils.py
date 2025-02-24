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


def split_data(x, y, device, train_ratio=(.6, .2, .2)):
    train_cnt = int(x.size(0) * train_ratio[0])
    valid_cnt = int(x.size(0) * train_ratio[1])
    test_cnt = int(x.size(0) * train_ratio[2])
    cnt = [train_cnt, valid_cnt, test_cnt]

    print(cnt)

    # shuffle
    indices = torch.randperm(x.size(0)).to(device)
    x = torch.index_select(x, dim=0, index=indices)
    y = torch.index_select(y, dim=0, index=indices)

    print(x.shape)
    print(y.shape)

    # Split
    x = list(x.split(cnt, dim=0))
    y = list(y.split(cnt, dim=0))

    return x, y