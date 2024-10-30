import torch
import os

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_project_root() -> str:
    """Return the absolute path to the project root."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != os.path.dirname(current_dir):  # Root has the same parent directory
        if '.root' in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return current_dir


def prefix_path_with_root(path):
    root_path = get_project_root()
    return root_path + "/" + path