from typing import List, TYPE_CHECKING
import torch
from src.constants import WEIGHTS_PRUNING_ATTR, WEIGHTS_ATTR, WEIGHTS_FLIPPING_ATTR
if TYPE_CHECKING:
    from src.layers import  LayerPrimitive

from src.others import get_device
from torch import nn

def get_parameters_flipped_statistics_sigmoid_(layer_primitive: 'LayerPrimitive') -> tuple[float, float]:
    total = 0
    remaining_non_flipped = 0

    weights = getattr(layer_primitive, WEIGHTS_ATTR)
    mask_pruning = getattr(layer_primitive, WEIGHTS_FLIPPING_ATTR)

    total += weights.numel()
    remaining_non_flipped += (torch.sigmoid(mask_pruning) >= 0.5).float().sum()
    return total, remaining_non_flipped

def get_parameters_pruning_statistics_vanilla_(layer_primitive: 'LayerPrimitive') -> tuple[float, float]:
    total = 0
    remaining = 0

    weights = getattr(layer_primitive, WEIGHTS_ATTR)
    total += weights.numel()
    remaining += (weights != 0).float().sum()
    return total, remaining

def get_parameters_pruning_statistics_sigmoid_(layer_primitive: 'LayerPrimitive') -> tuple[float, float]:
    total = 0
    remaining = 0

    weights = getattr(layer_primitive, WEIGHTS_ATTR)
    mask_pruning = getattr(layer_primitive, WEIGHTS_PRUNING_ATTR)

    total += weights.numel()
    remaining += (torch.sigmoid(mask_pruning) >= 0.5).float().sum()
    return total, remaining

def get_parameters_pruning_sigmoid_steep(layer_primitive: 'LayerPrimitive') -> tuple[float, torch.Tensor]:
    total = 0
    sigmoids = torch.tensor(0, device=get_device(), dtype=torch.float)

    weights = getattr(layer_primitive, WEIGHTS_ATTR)
    mask_pruning = getattr(layer_primitive, WEIGHTS_PRUNING_ATTR)

    total += weights.numel()
    sigmoids += torch.sigmoid(50 * mask_pruning).sum()
    return total, sigmoids

def get_parameters_total(layer_primitive: 'LayerPrimitive') -> int:
    total = 0
    weights = getattr(layer_primitive, WEIGHTS_ATTR)
    total += weights.numel()
    return total

def get_parameters_pruning_sigmoid_(layer_primitive: 'LayerPrimitive') -> tuple[float, torch.Tensor]:
    total = 0
    sigmoids = torch.tensor(0, device=get_device(), dtype=torch.float)

    weights = getattr(layer_primitive, WEIGHTS_ATTR)
    mask_pruning = getattr(layer_primitive, WEIGHTS_PRUNING_ATTR)

    total += weights.numel()
    sigmoids += torch.sigmoid(mask_pruning).sum()
    return total, sigmoids

def get_parameters_flipped_statistics_leaky(layer_primitive: 'LayerPrimitive') -> tuple[float, float]:
    total = 0
    remaining_non_flipped = 0

    weights = getattr(layer_primitive, WEIGHTS_ATTR)
    mask_pruning = getattr(layer_primitive, WEIGHTS_FLIPPING_ATTR)

    total += weights.numel()
    #torch.nn.functional.leaky_relu(mask_param, negative_slope=0.01)
    remaining_non_flipped += (torch.nn.functional.leaky_relu(mask_pruning, negative_slope= 0.01) >= 0).float().sum()
    return total, remaining_non_flipped

def get_parameters_pruning_statistics_leaky(layer_primitive: 'LayerPrimitive') -> tuple[float, float]:
    total = 0
    remaining = 0

    weights = getattr(layer_primitive, WEIGHTS_ATTR)
    mask_pruning = getattr(layer_primitive, WEIGHTS_PRUNING_ATTR)

    total += weights.numel()
    remaining += (torch.nn.functional.leaky_relu(mask_pruning, negative_slope= 0.01) >= 0).float().sum()
    return total, remaining

def get_parameters_pruning_leaky(layer_primitive: 'LayerPrimitive') -> tuple[float, torch.Tensor]:
    total = 0
    leakys = torch.tensor(0, device=get_device(), dtype=torch.float)

    weights = getattr(layer_primitive, WEIGHTS_ATTR)
    mask_pruning = getattr(layer_primitive, WEIGHTS_PRUNING_ATTR)

    total += weights.numel()
    leakys += torch.nn.functional.leaky_relu(mask_pruning, negative_slope= 0).sum()

    return total, leakys