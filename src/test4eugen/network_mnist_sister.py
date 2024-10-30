import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from src.common import ConfigsNetworkMasks, LayerLinear, MaskPruningFunction, MaskFlipFunction, ConfigsLinear
from src.utils import get_device
from src.constants import MASK_PRUNING_ATTR, MASK_FLIPPING_ATTR, WEIGHTS_SISTER_ATTR, WEIGHTS_ATTR, BIAS_ATTR, \
    MASK_PRUNING_SISTER_ATTR


class LayerLinearSister(nn.Module):
    def __init__(self, configs_linear: ConfigsLinear, configs_network: ConfigsNetworkMasks):
        super(LayerLinearSister, self).__init__()

        self.in_features = configs_linear['in_features']
        self.out_features = configs_linear['out_features']

        self.mask_pruning_enabled = configs_network['mask_pruning_enabled']
        self.weights_training_enabled = configs_network['weights_training_enabled']
        self.mask_flipping_enabled = configs_network['mask_flipping_enabled']


        setattr(self, WEIGHTS_ATTR, nn.Parameter(torch.Tensor(self.out_features, self.in_features)))
        setattr(self, WEIGHTS_SISTER_ATTR, nn.Parameter(torch.Tensor(self.out_features, self.in_features)))
        setattr(self, BIAS_ATTR, nn.Parameter(torch.Tensor(self.out_features)))

        getattr(self, WEIGHTS_ATTR).requires_grad = self.weights_training_enabled
        getattr(self, WEIGHTS_SISTER_ATTR).requires_grad = self.weights_training_enabled
        getattr(self, BIAS_ATTR).requires_grad = self.weights_training_enabled

        setattr(self, MASK_PRUNING_ATTR,  nn.Parameter(torch.Tensor(self.out_features, self.in_features)))
        setattr(self, MASK_PRUNING_SISTER_ATTR,  nn.Parameter(torch.Tensor(self.out_features, self.in_features)))
        getattr(self, MASK_PRUNING_ATTR).requires_grad = self.mask_pruning_enabled
        getattr(self, MASK_PRUNING_SISTER_ATTR).requires_grad = self.mask_pruning_enabled

        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_uniform_(getattr(self, WEIGHTS_ATTR), a=math.sqrt(5))
        getattr(self, WEIGHTS_ATTR).data = torch.abs(getattr(self, WEIGHTS_ATTR).data)
        nn.init.kaiming_uniform_(getattr(self, WEIGHTS_SISTER_ATTR), a=math.sqrt(5))
        getattr(self, WEIGHTS_SISTER_ATTR).data = -torch.abs(getattr(self, WEIGHTS_SISTER_ATTR).data)

        nn.init.uniform_(getattr(self, MASK_PRUNING_ATTR), a=1, b=1)
        nn.init.uniform_(getattr(self, MASK_PRUNING_SISTER_ATTR), a=1, b=1)

        weights = getattr(self, WEIGHTS_ATTR)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(getattr(self, BIAS_ATTR), -bound, bound)

    def forward(self, input):
        masked_weight = getattr(self, WEIGHTS_ATTR)
        masked_weight_sister = getattr(self, WEIGHTS_SISTER_ATTR)
        bias = getattr(self, BIAS_ATTR)

        if self.mask_pruning_enabled:
            mask_changes = MaskPruningFunction.apply(getattr(self, MASK_PRUNING_ATTR))
            masked_weight = masked_weight * mask_changes

            mask_changes_sister = MaskPruningFunction.apply(getattr(self, MASK_PRUNING_SISTER_ATTR))
            masked_weight_sister = masked_weight_sister * mask_changes_sister

        return F.linear(input, masked_weight + masked_weight_sister, bias)



class ModelMnistFNNSister(nn.Module):
    def __init__(self, config_network_mask: ConfigsNetworkMasks):
        super(ModelMnistFNNSister, self).__init__()
        self.fc1 = LayerLinearSister(
            configs_linear={'in_features': 28*28, 'out_features': 300},
            configs_network=config_network_mask
        )
        self.fc2 = LayerLinearSister(
            configs_linear={'in_features': 300, 'out_features': 100},
            configs_network=config_network_mask
        )
        self.fc3 = LayerLinearSister(
            configs_linear={'in_features': 100, 'out_features': 10},
            configs_network=config_network_mask
        )

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_masked_loss(self) -> torch.Tensor:
        total = 0
        masked = torch.tensor(0, device=get_device(), dtype=torch.float)
        for layer in [self.fc1, self.fc2, self.fc3]:
            total += layer.weights.numel()
            mask = torch.sigmoid(getattr(layer, MASK_PRUNING_ATTR))
            # Apply threshold at 0.5 to get binary mask
            mask_sister = torch.sigmoid(getattr(layer, MASK_PRUNING_SISTER_ATTR))
            masked += mask.sum() + mask_sister.sum()

        return masked / total

    def get_masked_percentage(self) -> float:
        total = 0
        masked = torch.tensor(0, device=get_device(), dtype=torch.float)
        for layer in [self.fc1, self.fc2, self.fc3]:
            total += layer.weights.numel()
            mask = torch.sigmoid(getattr(layer, MASK_PRUNING_ATTR))
            # Apply threshold at 0.5 to get binary mask
            mask_thresholded = (mask >= 0.5).float()

            mask_sister = torch.sigmoid(getattr(layer, MASK_PRUNING_SISTER_ATTR))

            mask_thresholded_sister = (mask_sister >= 0.5).float()
            masked += mask_thresholded.sum() + mask_thresholded_sister.sum()

        return masked.item() / total
