from typing import List
import torch
from src.constants import MASK_PRUNING_ATTR
from src.utils import get_device
from torch import nn

def get_pruning_loss(model: nn.Module, registered_layers: List[any]) -> torch.Tensor:
    total = 0
    masked = torch.tensor(0, device=get_device(), dtype=torch.float)
    for layer in registered_layers:
        total += layer.weights.numel()
        mask = torch.sigmoid(getattr(layer, MASK_PRUNING_ATTR))
        masked += mask.sum()

    return masked / total
