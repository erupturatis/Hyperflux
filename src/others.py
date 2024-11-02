import torch
import os


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_root_folder() -> str:
    """Return the absolute path to the project root."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != os.path.dirname(current_dir):  # Root has the same parent directory
        if '.root' in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return current_dir


def prefix_path_with_root(path):
    root_path = get_root_folder()
    return root_path + "/" + path

def balancer_parameters(network_loss: float, regularization_loss: float, scale:float = 1, ratio: float = 1) -> tuple[float, float]:
    """
    Balances the network losses in the desired ratio
    :param network_loss: ...
    :param regularization_loss: ...
    :param scale: represents the value of the final network loss
    :param ratio: represents the ratio: regularization_loss / network_loss
    :return:
    """
    a = scale / network_loss
    b = (a * ratio * network_loss) / regularization_loss
    return a, b

