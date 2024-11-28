import torch
from typing import List
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from src.layers import ConfigsNetworkMasksImportance, LayerLinearMaskImportance, MaskPruningFunctionSigmoid, \
    MaskFlipFunctionSigmoid, ConfigsLayerLinear, \
    get_remaining_parameters_loss_masks_importance, get_layer_composite_flipped_statistics, \
    get_layer_composite_pruning_statistics, \
    LayerPrimitive, LayerComposite, get_layers_primitive, LayerLinearMaskProbabilitiesPruneSign, \
    ConfigsNetworkMasksProbabilitiesPruneSign, get_parameters_probabilities_statistics, \
    get_layer_composite_pruning_statistics_probabilities
from src.others import get_device
from src.constants import WEIGHTS_PRUNING_ATTR, WEIGHTS_FLIPPING_ATTR

class ModelMnistFNNProbabilistic(LayerComposite):
    def __init__(self, config_network_mask: ConfigsNetworkMasksProbabilitiesPruneSign):
        super(ModelMnistFNNProbabilistic, self).__init__()
        self.fc1 = LayerLinearMaskProbabilitiesPruneSign(
            configs_linear=ConfigsLayerLinear(in_features=28*28, out_features=300),
            configs_network=config_network_mask
        )
        self.fc2 = LayerLinearMaskProbabilitiesPruneSign(
            configs_linear=ConfigsLayerLinear(in_features=300, out_features=100),
            configs_network=config_network_mask
        )
        self.fc3 = LayerLinearMaskProbabilitiesPruneSign(
            configs_linear=ConfigsLayerLinear(in_features=100, out_features=10),
            configs_network=config_network_mask
        )
        self.registered_layers = [self.fc1, self.fc2, self.fc3]

    def get_remaining_parameters_loss(self) -> torch.Tensor:
        pass
        # total, sigmoid =  get_remaining_parameters_loss_masks_importance(self)
        # return sigmoid / total

    def get_layers_primitive(self) -> List[LayerPrimitive]:
        return get_layers_primitive(self)

    def get_parameters_statistics(self) -> tuple[int, int, int]:
        return get_layer_composite_pruning_statistics_probabilities(self)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_inference(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = F.relu(self.fc1.forward_inference(x))
        x = F.relu(self.fc2.forward_inference(x))
        x = self.fc3.forward_inference(x)
        return x

