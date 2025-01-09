import torch
from typing import List
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from fontTools.config import Config

from src.layers import ConfigsNetworkMasksImportance, LayerLinearMaskImportance, MaskPruningFunctionSigmoid, MaskFlipFunctionSigmoid, ConfigsLayerLinear, \
    get_remaining_parameters_loss_masks_importance, get_layer_composite_flipped_statistics, get_layer_composite_pruning_statistics, \
    LayerPrimitive, LayerComposite, get_layers_primitive
from src.others import get_device
from src.constants import WEIGHTS_PRUNING_ATTR, WEIGHTS_FLIPPING_ATTR

class ModelMnistFNN(LayerComposite):
    def __init__(self, config_network_mask: ConfigsNetworkMasksImportance):
        super(ModelMnistFNN, self).__init__()
        self.fc1 = LayerLinearMaskImportance(
            configs_linear=ConfigsLayerLinear(in_features=28*28, out_features=300),
            configs_network=config_network_mask
        )
        self.fc2 = LayerLinearMaskImportance(
            configs_linear=ConfigsLayerLinear(in_features=300, out_features=100),
            configs_network=config_network_mask
        )
        self.fc3 = LayerLinearMaskImportance(
            configs_linear=ConfigsLayerLinear(in_features=100, out_features=10),
            configs_network=config_network_mask
        )

        # self.registered_layers = [self.fc3]
        self.registered_layers = [self.fc1, self.fc2, self.fc3]



    def get_remaining_parameters_loss(self) -> torch.Tensor:
        total, sigmoid =  get_remaining_parameters_loss_masks_importance(self)
        return sigmoid / total

    def get_layers_primitive(self) -> List[LayerPrimitive]:
        return get_layers_primitive(self)

    def get_parameters_pruning_statistics(self) -> any:
        return get_layer_composite_pruning_statistics(self)

    def get_parameters_flipped_statistics(self) -> any:
        return get_layer_composite_flipped_statistics(self)

    def forward(self, x, inference = False):
        x = x.view(-1, 28*28)  # Flatten the input
        x = F.relu(self.fc1(x, inference=inference))
        x = F.relu(self.fc2(x, inference=inference))
        x = self.fc3(x, inference=inference)
        return x

class ModelMnistFNNAllToAll(LayerComposite):
    def __init__(self, config_network_mask: ConfigsNetworkMasksImportance):
        super(ModelMnistFNNAllToAll, self).__init__()

        # Define the primary layers
        self.fc1 = LayerLinearMaskImportance(
            configs_linear=ConfigsLayerLinear(in_features=28*28, out_features=300),
            configs_network=config_network_mask
        )
        self.fc2 = LayerLinearMaskImportance(
            configs_linear=ConfigsLayerLinear(in_features=300, out_features=100),
            configs_network=config_network_mask
        )
        self.fc3 = LayerLinearMaskImportance(
            configs_linear=ConfigsLayerLinear(in_features=100, out_features=10),
            configs_network=config_network_mask
        )

        # Define additional transformation layers for multi-input connections
        # For fc2: inputs from input layer and fc1
        self.fc2_input_transform = LayerLinearMaskImportance(
            configs_linear=ConfigsLayerLinear(in_features=28*28, out_features=100),
            configs_network=config_network_mask
        )
        self.fc2_fc1_transform = LayerLinearMaskImportance(
            configs_linear=ConfigsLayerLinear(in_features=300, out_features=100),
            configs_network=config_network_mask
        )

        # For fc3: inputs from input layer, fc1, and fc2
        self.fc3_input_transform = LayerLinearMaskImportance(
            configs_linear=ConfigsLayerLinear(in_features=28*28, out_features=10),
            configs_network=config_network_mask
        )
        self.fc3_fc1_transform = LayerLinearMaskImportance(
            configs_linear=ConfigsLayerLinear(in_features=300, out_features=10),
            configs_network=config_network_mask
        )
        self.fc3_fc2_transform = LayerLinearMaskImportance(
            configs_linear=ConfigsLayerLinear(in_features=100, out_features=10),
            configs_network=config_network_mask
        )

        # Register all layers
        self.registered_layers = [
            self.fc1, self.fc2, self.fc3,
            self.fc2_input_transform, self.fc2_fc1_transform,
            self.fc3_input_transform, self.fc3_fc1_transform, self.fc3_fc2_transform
        ]

    def get_remaining_parameters_loss(self) -> torch.Tensor:
        total, sigmoid = get_remaining_parameters_loss_masks_importance(self)
        return sigmoid / total

    def get_layers_primitive(self) -> List[LayerPrimitive]:
        return get_layers_primitive(self)

    def get_parameters_pruning_statistics(self) -> any:
        return get_layer_composite_pruning_statistics(self)

    def get_parameters_flipped_statistics(self) -> any:
        return get_layer_composite_flipped_statistics(self)

    def forward(self, x, inference=False):
        x = x.view(-1, 28*28)  # Flatten the input

        # Pass through fc1
        out1 = F.relu(self.fc1(x, inference=inference))

        # Compute fc2 outputs from input and fc1
        fc2_from_input = self.fc2_input_transform(x, inference=inference)
        fc2_from_fc1 = self.fc2_fc1_transform(out1, inference=inference)
        out2 = F.relu(fc2_from_input + fc2_from_fc1)

        # Compute fc3 outputs from input, fc1, and fc2
        fc3_from_input = self.fc3_input_transform(x, inference=inference)
        fc3_from_fc1 = self.fc3_fc1_transform(out1, inference=inference)
        fc3_from_fc2 = self.fc3_fc2_transform(out2, inference=inference)
        out3 = fc3_from_input + fc3_from_fc1 + fc3_from_fc2  # No activation here (e.g., for logits)

        return out3

class ModelMnistFNNAllToAllOther(LayerComposite):
    def __init__(self, config_network_mask: ConfigsNetworkMasksImportance):
        super(ModelMnistFNNAllToAllOther, self).__init__()

        self.fc1 = LayerLinearMaskImportance(
            configs_linear=ConfigsLayerLinear(in_features=28*28, out_features=300),
            configs_network=config_network_mask
        )
        self.fc2 = LayerLinearMaskImportance(
            configs_linear=ConfigsLayerLinear(in_features=300 + 28*28, out_features=100),
            configs_network=config_network_mask
        )
        self.fc3 = LayerLinearMaskImportance(
            configs_linear=ConfigsLayerLinear(in_features=100 + 300 + 28*28, out_features=10),
            configs_network=config_network_mask
        )

        self.registered_layers = [
            self.fc1, self.fc2, self.fc3,
        ]

    def get_remaining_parameters_loss(self) -> torch.Tensor:
        total, sigmoid = get_remaining_parameters_loss_masks_importance(self)
        return sigmoid / total

    def get_layers_primitive(self) -> List[LayerPrimitive]:
        return get_layers_primitive(self)

    def get_parameters_pruning_statistics(self) -> any:
        return get_layer_composite_pruning_statistics(self)

    def get_parameters_flipped_statistics(self) -> any:
        return get_layer_composite_flipped_statistics(self)

    def forward(self, x, inference=False):
        x = x.view(-1, 28*28)  # Flatten the input

        # Pass through fc1
        out1 = F.relu(self.fc1(x, inference=inference))

        # Concatenate input and fc1 output, then pass through fc2
        concatenated1 = torch.cat((x, out1), dim=1)
        out2 = F.relu(self.fc2(concatenated1, inference=inference))

        # Concatenate input, fc1 output, and fc2 output, then pass through fc3
        concatenated2 = torch.cat((x, out1, out2), dim=1)
        out3 = self.fc3(concatenated2, inference=inference)

        return out3
