import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import TypedDict

from src.common import ConfigsNetworkMasks, LayerLinear, MaskFlipFunction, MaskPruningFunction, ConfigsLinear
from src.utils import get_device
import math
import numpy as np

from src.variables import WEIGHTS_ATTR, BIAS_ATTR, MASK_FLIPPING_ATTR, MASK_PRUNING_ATTR


class ConfigsConv2D(TypedDict):
    in_channels: int
    out_channels: int
    kernel_size: int
    padding: int
    stride: int


class LayerConv2(nn.Module):
    def __init__(self, configs_conv2d: ConfigsConv2D, configs_network_masks: ConfigsNetworkMasks):
        super(LayerConv2, self).__init__()
        # getting configs
        self.in_channels = configs_conv2d['in_channels']
        self.out_channels = configs_conv2d['out_channels']
        self.kernel_size = configs_conv2d['kernel_size']
        self.padding = configs_conv2d['padding']
        self.stride = configs_conv2d['stride']

        self.mask_pruning_enabled = configs_network_masks['mask_pruning_enabled']
        self.mask_flipping_enabled = configs_network_masks['mask_flipping_enabled']
        self.weights_training_enabled = configs_network_masks['weights_training_enabled']

        # defining parameters
        setattr(self, WEIGHTS_ATTR, nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)))
        setattr(self, MASK_PRUNING_ATTR, nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)))
        setattr(self, MASK_FLIPPING_ATTR, nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)))
        setattr(self, BIAS_ATTR, nn.Parameter(torch.Tensor(self.out_channels)))

        # turning on and off params
        getattr(self, WEIGHTS_ATTR).requires_grad = self.weights_training_enabled
        getattr(self, BIAS_ATTR).requires_grad = self.weights_training_enabled
        getattr(self, MASK_PRUNING_ATTR).requires_grad = self.mask_pruning_enabled
        getattr(self, MASK_FLIPPING_ATTR).requires_grad = self.mask_flipping_enabled

        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_uniform_(getattr(self, WEIGHTS_ATTR), a=math.sqrt(5))
        nn.init.uniform_(getattr(self, MASK_PRUNING_ATTR), a=0.2, b=0.2)
        nn.init.uniform_(getattr(self, MASK_FLIPPING_ATTR), a=0.2, b=0.2)

        weights = getattr(self, WEIGHTS_ATTR)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(getattr(self, BIAS_ATTR), -bound, bound)

    def forward(self, input):
        masked_weights = getattr(self, WEIGHTS_ATTR)
        bias = getattr(self, BIAS_ATTR)
        mask_changes = None

        if self.mask_pruning_enabled:
            mask_changes = MaskPruningFunction.apply(getattr(self, MASK_PRUNING_ATTR))
            masked_weights = masked_weights * mask_changes

        if self.mask_flipping_enabled:
            mask_changes = MaskFlipFunction.apply(getattr(self, MASK_FLIPPING_ATTR))
            masked_weights = masked_weights * mask_changes

        return F.conv2d(input, masked_weights, bias, self.stride, self.padding)



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
            mask = torch.sigmoid(layer.mask_pruning)
            # Apply threshold at 0.5 to get binary mask
            masked += mask.sum()

        return masked / total

    def get_masked_percentage(self) -> float:
        total = 0
        masked = torch.tensor(0, device=get_device(), dtype=torch.float)
        for layer in [self.fc1, self.fc2, self.fc3, self.conv2D_1, self.conv2D_2]:
            total += layer.weights.numel()
            mask = torch.sigmoid(layer.mask_pruning)
            # Apply threshold at 0.5 to get binary mask
            mask_thresholded = (mask >= 0.5).float()
            masked += mask_thresholded.sum()

        return masked.item() / total

    def forward(self, x):

        x = F.relu(self.conv2D_1(x))
        x = F.relu(self.conv2D_2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x
