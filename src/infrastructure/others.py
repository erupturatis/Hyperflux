from calendar import month
import torch
import os

from src.infrastructure.constants import SAVED_RESULTS_PATH


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

def round_float(value: float, digits: int = 4) -> float:
    return round(value, digits)

import json
def save_array_experiment(filename, arr):
    path = prefix_path_with_root(SAVED_RESULTS_PATH)
    path = path + "/" + filename
    with open(path, 'w') as file:
        json.dump(arr, file)


def get_model_sparsity_percent(model) -> float:
    total, remaining = model.get_parameters_pruning_statistics()
    percent = remaining / total * 100
    return round_float(percent.item())
