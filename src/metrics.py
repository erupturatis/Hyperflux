from typing import List
import torch
from src.constants import WEIGHTS_PRUNING_ATTR, WEIGHTS_FLIPPING_ATTR
from src.others import get_device
from torch import nn
#
# def get_pruned_percentage(model: nn.Module, registered_layers: List[any]) -> float:
#     total = 0
#     masked = torch.tensor(0, device=get_device(), dtype=torch.float)
#     for layer in registered_layers:
#         total += layer.weights.numel()
#         mask = torch.sigmoid(getattr(layer, MASK_PRUNING_ATTR))
#
#         mask_threshold = (mask >= 0.5).float()
#         masked += mask_threshold.sum()
#
#     return masked.item() / total
#
# def get_flipped_percentage(model: nn.Module, registered_layers: List[any]) -> float:
#     total = 0
#     flipped = torch.tensor(0, device=get_device(), dtype=torch.float)
#     for layer in registered_layers:
#         total += layer.weights.numel()
#         mask = torch.sigmoid(getattr(layer, MASK_FLIPPING_ATTR))
#
#         mask_threshold = (mask >= 0.5).float()
#         flipped += mask_threshold.sum()
#
#     return (total-flipped.item()) / total
