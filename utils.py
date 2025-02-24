import pandas as pd

import torch

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

def load_data():
    california = fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)

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

    # shuffle
    indices = torch.randperm(x.size(0))
    x = torch.index_select(x, dim=0, index=indices)
    y_ = torch.index_select(y, dim=0, index=indices)

    # Split
    x = list(x.split(cnt, dim=0))
    y = list(y_.split(cnt, dim=0))

    # Standardization - mean = 0, variance = 1
    scaler = StandardScaler()
    scaler.fit(x[0].numpy())
    x[0] = torch.from_numpy(scaler.transform(x[0].numpy())).float()
    x[1] = torch.from_numpy(scaler.transform(x[1].numpy())).float()
    x[2] = torch.from_numpy(scaler.transform(x[2].numpy())).float()

    # torch to device
    for idx in range(len(x)):
        x[idx] = x[idx].to(device)
        y[idx] = y[idx].to(device)

    return x, y