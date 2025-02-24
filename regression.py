import pandas as pd
import torch

from model import MyDNNModel

from utils import load_data
from utils import split_data

def load(model_fn, device):
    d = torch.load(model_fn, map_location=device, weights_only=False)

    return d["model"], d["config"]

def test(model, x, y, to_be_shown=False):
    model.eval()

    with torch.no_grad():
        y_hat = model(x)

        if to_be_shown:
            import seaborn as sns
            import matplotlib.pyplot as plt

            y, y_hat = y.to("cpu"), y_hat.to("cpu")

            df = pd.DataFrame(torch.cat([y, y_hat], dim=1).detach_().numpy(), columns=['y', "y_hat"])
            sns.pairplot(df, height=5)
            plt.show()

def main():
    # Model weight file path
    model_fn = "./model/model.pth"

    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Define Model
    model_dict, config = load(model_fn, device)

    # Load Data
    x, y = load_data()
    x, y = split_data(x, y, device, config["train_ratio"])

    test_x, test_y = x[2], y[2]

    model = MyDNNModel(input_size=test_x.size(-1), output_size=test_y.size(-1)).to(device)
    model.load_state_dict(model_dict)

    # Test
    test(model, test_x, test_y, to_be_shown=True)

if __name__ == "__main__":
    main()