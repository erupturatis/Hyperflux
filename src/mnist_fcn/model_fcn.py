import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from src.layers import ConfigsNetworkMasks, LayerLinear, MaskPruningFunction, MaskFlipFunction
from src.metrics import get_pruned_percentage, get_flipped_percentage
from src.utils import get_device
from src.constants import MASK_PRUNING_ATTR, MASK_FLIPPING_ATTR

class ModelMnistFNN(nn.Module):
    def __init__(self, config_network_mask: ConfigsNetworkMasks):
        super(ModelMnistFNN, self).__init__()
        self.fc1 = LayerLinear(
            configs_linear={'in_features': 28*28, 'out_features': 300},
            configs_network=config_network_mask
        )
        self.fc2 = LayerLinear(
            configs_linear={'in_features': 300, 'out_features': 100},
            configs_network=config_network_mask
        )
        self.fc3 = LayerLinear(
            configs_linear={'in_features': 100, 'out_features': 10},
            configs_network=config_network_mask
        )
        self.registered_layers = [self.fc1, self.fc2, self.fc3]


    def get_pruning_loss(self) -> torch.Tensor:
        return get_pruning_loss(self, self.registered_layers)

    def get_pruned_percentage(self) -> float:
        return get_pruned_percentage(self, self.registered_layers)

    def get_flipped_percentage(self) -> float:
        return get_flipped_percentage(self, self.registered_layers)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

