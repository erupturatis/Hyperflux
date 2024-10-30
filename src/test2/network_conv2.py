import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import TypedDict

from src.common import ConfigsNetworkMasks, LayerLinear, MaskFlipFunction, MaskPruningFunction, ConfigsLinear, ConfigsConv2D, LayerConv2
from src.utils import get_device
import math
import numpy as np

from src.variables import WEIGHTS_ATTR, BIAS_ATTR, MASK_FLIPPING_ATTR, MASK_PRUNING_ATTR




class ModelCifar10Conv2(nn.Module):
    def __init__(self, configs_network_masks: ConfigsNetworkMasks):
        super(ModelCifar10Conv2, self).__init__()

        configs_conv2d_1: ConfigsConv2D = {
            'in_channels': 3,
            'out_channels': 64,
            'kernel_size': 3,
            'padding': 1,
            'stride': 1
        }
        self.conv2D_1 = LayerConv2(configs_conv2d_1, configs_network_masks)

        configs_conv2d_2: ConfigsConv2D = {
            'in_channels': 64,
            'out_channels': 64,
            'kernel_size': 3,
            'padding': 1,
            'stride': 1
        }
        self.conv2D_2 = LayerConv2(configs_conv2d_2, configs_network_masks)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        
        config_linear_1: ConfigsLinear = {
            'in_features': 64 * 16 * 16,
            'out_features': 256
        }
        self.fc1 = LayerLinear(config_linear_1, configs_network_masks)
        config_linear_2: ConfigsLinear = {
            'in_features': 256,
            'out_features': 256
        }
        self.fc2 = LayerLinear(config_linear_2, configs_network_masks)
        config_linear_3: ConfigsLinear = {
            'in_features': 256,
            'out_features': 10
        }
        self.fc3 = LayerLinear(config_linear_3, configs_network_masks)



    def get_masked_loss(self) -> torch.Tensor:
        total = 0
        masked = torch.tensor(0, device=get_device(), dtype=torch.float)
        for layer in [self.fc1, self.fc2, self.fc3, self.conv2D_1, self.conv2D_2]:
            total += layer.weights.numel()
            mask = torch.sigmoid(getattr(layer, MASK_PRUNING_ATTR))
            # Apply threshold at 0.5 to get binary mask
            masked += mask.sum()

        return masked / total

    def get_masked_percentage(self) -> float:
        total = 0
        masked = torch.tensor(0, device=get_device(), dtype=torch.float)
        for layer in [self.fc1, self.fc2, self.fc3, self.conv2D_1, self.conv2D_2]:
            total += layer.weights.numel()
            mask = torch.sigmoid(getattr(layer, MASK_PRUNING_ATTR))
            # Apply threshold at 0.5 to get binary mask
            mask_thresholded = (mask >= 0.5).float()
            masked += mask_thresholded.sum()

        return masked.item() / total

    def get_flipped_percentage(self) -> float:
        total = 0
        flipped = torch.tensor(0, device=get_device(), dtype=torch.float)
        for layer in [self.fc1, self.fc2, self.fc3, self.conv2D_1, self.conv2D_2]:
            total += layer.weights.numel()
            mask = torch.sigmoid(getattr(layer, MASK_FLIPPING_ATTR))
            # Apply threshold at 0.5 to get binary mask
            mask_thresholded = (mask >= 0.5).float()
            flipped += mask_thresholded.sum()

        return (total-flipped.item()) / total

    def forward(self, x):

        x = F.relu(self.conv2D_1(x))
        x = F.relu(self.conv2D_2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x
