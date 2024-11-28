import torch
from typing import List
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import TypedDict
from src.layers import ConfigsNetworkMasksImportance, LayerLinearMaskImportance, ConfigsLayerLinear, \
    ConfigsLayerConv2, LayerConv2MaskImportance, LayerComposite, LayerPrimitive, get_remaining_parameters_loss_masks_importance, get_layers_primitive, \
    get_layer_composite_pruning_statistics, get_layer_composite_flipped_statistics, get_remaining_parameters_loss_steep, \
    get_parameters_total_count
from src.others import get_device
import math
import numpy as np
from src.constants import WEIGHTS_ATTR, BIAS_ATTR, WEIGHTS_FLIPPING_ATTR, WEIGHTS_PRUNING_ATTR

class ModelCifar10Conv2(LayerComposite):
    def __init__(self, configs_network_masks: ConfigsNetworkMasksImportance):
        super(ModelCifar10Conv2, self).__init__()


        configs_conv2d_1 = ConfigsLayerConv2(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.conv2D_1 = LayerConv2MaskImportance(configs_conv2d_1, configs_network_masks)

        configs_conv2d_2 = ConfigsLayerConv2(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.conv2D_2 = LayerConv2MaskImportance(configs_conv2d_2, configs_network_masks)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        config_linear_1 = ConfigsLayerLinear(in_features=64 * 16 * 16, out_features=256)
        self.fc1 = LayerLinearMaskImportance(config_linear_1, configs_network_masks)
        config_linear_2 = ConfigsLayerLinear(in_features=256, out_features=256)
        self.fc2 = LayerLinearMaskImportance(config_linear_2, configs_network_masks)
        config_linear_3 = ConfigsLayerLinear(in_features=256, out_features=10)
        self.fc3 = LayerLinearMaskImportance(config_linear_3, configs_network_masks)

        self.registered_layers = [self.fc1, self.fc2, self.fc3, self.conv2D_1, self.conv2D_2]

    def get_remaining_parameters_loss_steep(self) -> torch.Tensor:
        """
        Returns the loss with a steep sigmoid function, 50*x
        """
        total, sigmoid =  get_remaining_parameters_loss_steep(self)
        return sigmoid / total

    def get_remaining_parameters_loss(self) -> torch.Tensor:
        total, sigmoid =  get_remaining_parameters_loss_masks_importance(self)
        return sigmoid / total

    def get_parameters_total_count(self) -> int:
        total = get_parameters_total_count(self)
        return total

    def get_layers_primitive(self) -> List[LayerPrimitive]:
        return get_layers_primitive(self)

    def get_parameters_pruning_statistics(self) -> any:
        return get_layer_composite_pruning_statistics(self)

    def get_parameters_flipped_statistics(self) -> any:
        return get_layer_composite_flipped_statistics(self)

    def forward(self, x):
        x = F.relu(self.conv2D_1(x))
        x = F.relu(self.conv2D_2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x
