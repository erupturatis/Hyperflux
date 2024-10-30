import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import TypedDict
from src.layers import ConfigsNetworkMasks, LayerLinear, MaskFlipFunction, MaskPruningFunction, ConfigsLayerLinear, \
     LayerConv2, ConfigsLayerConv2
from src.constants import MASK_FLIPPING_ATTR
from src.losses import get_pruning_loss
from src.metrics import get_pruned_percentage, get_flipped_percentage
from src.utils import get_device
import math
import numpy as np

class ModelCifar10Conv4(nn.Module):
    def __init__(self, configs_network_masks: ConfigsNetworkMasks):
        super(ModelCifar10Conv4, self).__init__()

        configs_conv2d_1: ConfigsLayerConv2 = {
            'in_channels': 3,
            'out_channels': 64,
            'kernel_size': 3,
            'padding': 1,
            'stride': 1
        }
        self.conv2D_1 = LayerConv2(configs_conv2d_1, configs_network_masks)

        configs_conv2d_2: ConfigsLayerConv2 = {
            'in_channels': 64,
            'out_channels': 64,
            'kernel_size': 3,
            'padding': 1,
            'stride': 1
        }
        self.conv2D_2 = LayerConv2(configs_conv2d_2, configs_network_masks)

        configs_conv2d_3: ConfigsLayerConv2 = {
            'in_channels': 64,
            'out_channels': 128,
            'kernel_size': 3,
            'padding': 1,
            'stride': 1
        }
        self.conv2D_3 = LayerConv2(configs_conv2d_3, configs_network_masks)

        configs_conv2d_4: ConfigsLayerConv2 = {
            'in_channels': 128,
            'out_channels': 128,
            'kernel_size': 3,
            'padding': 1,
            'stride': 1
        }
        self.conv2D_4 = LayerConv2(configs_conv2d_4, configs_network_masks)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()


        config_linear_1: ConfigsLayerLinear = {
            'in_features': 128 * 8 * 8,
            'out_features': 256
        }
        self.fc1 = LayerLinear(config_linear_1, configs_network_masks)

        config_linear_2: ConfigsLayerLinear = {
            'in_features': 256,
            'out_features': 256
        }
        self.fc2 = LayerLinear(config_linear_2, configs_network_masks)

        config_linear_3: ConfigsLayerLinear = {
            'in_features': 256,
            'out_features': 10
        }
        self.fc3 = LayerLinear(config_linear_3, configs_network_masks)
        self.registered_layers = [self.fc1, self.fc2, self.fc3, self.conv2D_1, self.conv2D_2, self.conv2D_3, self.conv2D_4]


    def get_pruning_loss(self) -> torch.Tensor:
        return get_pruning_loss(self, self.registered_layers)

    def get_pruned_percentage(self) -> float:
        return get_pruned_percentage(self, self.registered_layers)

    def get_flipped_percentage(self) -> float:
        return get_flipped_percentage(self, self.registered_layers)

    def forward(self, x):
        x = F.relu(self.conv2D_1(x))
        x = F.relu(self.conv2D_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2D_3(x))
        x = F.relu(self.conv2D_4(x))
        x = self.maxpool(x)
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x



