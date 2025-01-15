from types import SimpleNamespace
from typing import TYPE_CHECKING
import torch
import torch.nn as nn
from src.infrastructure.constants import PRUNED_MODELS_PATH
from src.infrastructure.layers import LayerComposite, LayerPrimitive
from typing import List
from src.infrastructure.others import prefix_path_with_root
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from fontTools.config import Config

from src.mnist_lenet300.model_attributes import LENET300_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING, \
    LENET300_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING

def forward_pass_lenet300(self: 'LayerComposite', x: torch.Tensor, inference=False) -> torch.Tensor:
    x = x.view(-1, 28 * 28)
    x = F.relu(self.fc1(x, inference=inference))
    x = F.relu(self.fc2(x, inference=inference))
    x = self.fc3(x, inference=inference)
    return x

def save_model_weights_lenet300(model: 'LayerComposite', model_name: str, skip_array: List = []):
    filepath = PRUNED_MODELS_PATH + "/" + model_name
    filepath = prefix_path_with_root(filepath)
    state_dict = {}

    for mapping in LENET300_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING:
        custom_name = mapping['custom_name']
        standard_name = mapping['standard_name']
        if custom_name in skip_array:
            continue

        layer = getattr(model, custom_name, None)
        if layer is None:
            print(f"Layer '{custom_name}' not found in the model.")
            continue

        if isinstance(layer, LayerPrimitive):
            state_dict[standard_name] = layer.get_applied_weights().data.clone()
            if layer.get_bias_enabled():
                bias_name = standard_name.replace('.weight', '.bias')
                state_dict[bias_name] = layer.bias.data.clone()

        else:
            print(f"Unhandled layer type for layer '{custom_name}': {type(layer)}")

    torch.save(state_dict, filepath)
    print(f"Model weights saved to {filepath}.")

def load_model_weights_lenet300(model: 'LayerComposite', model_dict, skip_array: List = []):
    state_dict = model_dict

    for mapping in LENET300_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING:
        standard_name = mapping['standard_name']
        custom_name = mapping['custom_name']
        if custom_name in skip_array:
            continue

        layer = getattr(model, custom_name, None)
        if layer is None:
            print(f"Layer '{custom_name}' not found in the model.")
            continue

        if isinstance(layer, LayerPrimitive):
            layer.weights.data.copy_(state_dict[standard_name])
            if layer.get_bias_enabled():
                bias_name = standard_name.replace('.weight', '.bias')
                layer.bias.data.copy_(state_dict[bias_name])

        else:
            print(f"Unhandled layer type for layer '{custom_name}': {type(layer)}")

def load_model_weights_lenet300_from_path(model: 'LayerComposite', model_name: str, skip_array: List = []):
    filepath = PRUNED_MODELS_PATH + "/" + model_name
    filepath = prefix_path_with_root(filepath)
    state_dict = torch.load(filepath)
    load_model_weights_lenet300(model, state_dict, skip_array)

