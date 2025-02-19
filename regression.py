import pandas as pd
import torch

from model import MyDNNModel

from utils import load_data

def load(model_fn, device):
    d = torch.load(model_fn, map_location=device, weights_only=False)

    return d["model"], d["config"]

def test(model, x, y, to_be_shown=False):
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

    # Load Data
    x, y = load_data()

    # Define Model
    model_dict, config = load(model_fn, device)
    model = MyDNNModel(input_size=x.size(-1), output_size=y.size(-1)).to(device)
    model.load_state_dict(model_dict)

    # Test
    test(model, x.to(device), y.to(device), to_be_shown=True)

if __name__ == "__main__":
    main()