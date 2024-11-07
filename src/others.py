from calendar import month
import torch
import os
from dataclasses import dataclass
from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from src.layers import LayerComposite

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


@dataclass
class ArgsDisplayModelStatistics:
    BATCH_PRINT_RATE: int
    DATA_LENGTH: int
    average_loss_names: List[str]
    model: 'LayerComposite'

    average_loss_arr: List[torch.Tensor] = None
    current_batch_idx: int = 0
    epoch: int = 0
    batch_size: int = 128

def update_args_display_model_statistics(args: ArgsDisplayModelStatistics, average_loss_arr: List[torch.Tensor], current_batch_idx: int, epoch: int):
    args.average_loss_arr = average_loss_arr
    args.current_batch_idx = current_batch_idx
    args.epoch = epoch

def display_model_statistics(args: ArgsDisplayModelStatistics):
    print_rate = args.BATCH_PRINT_RATE
    total_data_len = args.DATA_LENGTH
    average_loss_arr = args.average_loss_arr
    average_loss_names = args.average_loss_names
    model = args.model

    current_batch_idx = args.current_batch_idx
    BATCH_SIZE = args.batch_size
    epoch = args.epoch

    loss_mean_arr = [loss / print_rate for loss in average_loss_arr]

    total, remaining = model.get_parameters_pruning_statistics()
    percent = remaining / total * 100

    print(f"Train Epoch: {epoch} [{(current_batch_idx+1) * BATCH_SIZE}/{total_data_len}]")
    print(f"Masked weights percentage: {percent:.2f}% Flipped weights percentage: {percent:.2f}%")
    losses_str = " ".join([f" {average_loss_names[idx]}: {loss}" for idx, loss in enumerate(loss_mean_arr)])
    print(f"Losses: {losses_str}")

    args.average_loss_arr = []

def get_model_remaining_parameters_percentage(model: 'LayerComposite') -> float:
    total, remaining = model.get_parameters_pruning_statistics()
    return remaining / total
