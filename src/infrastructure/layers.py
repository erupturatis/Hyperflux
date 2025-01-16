from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import TypedDict
from src.infrastructure.configs_layers import MaskPruningFunction, MaskFlipFunction, get_parameters_pruning, \
    get_parameters_pruning_statistics,  configs_get_layers_initialization
from src.infrastructure.parameters_mask_processors import get_parameters_pruning_sigmoid_steep, get_parameters_total
from src.infrastructure.mask_functions import MaskPruningFunctionSigmoid, MaskPruningFunctionSigmoidDebugging
from src.infrastructure.others import get_device
import math
from src.infrastructure.constants import WEIGHTS_ATTR, BIAS_ATTR, WEIGHTS_PRUNING_ATTR, WEIGHTS_FLIPPING_ATTR, \
    WEIGHTS_BASE_ATTR, get_flow_params_init


class ConfigsNetworkMasksImportance(TypedDict):
    mask_pruning_enabled: bool
    weights_training_enabled: bool


@dataclass
class ConfigsLayerLinear:
    in_features: int
    out_features: int
    bias_enabled: bool = True

class LayerPrimitive(nn.Module, ABC):
    # @abstractmethod
    # def get_remaining_parameters_loss(self) -> torch.Tensor:
    #     pass

    # @abstractmethod
    # def get_pruned_percentage(self) -> float:
    #     pass
    #
    # @abstractmethod
    # def get_flipped_percentage(self) -> float:
    #     pass
    pass

class LayerComposite(nn.Module, ABC):
    @abstractmethod
    def get_layers_primitive(self) -> List[LayerPrimitive]:
        pass

    # @abstractmethod
    # def get_remaining_parameters_loss(self) -> any:
    #     pass
    #
    # @abstractmethod
    # def get_parameters_pruning_statistics(self) -> any:
    #     pass

def get_parameters_probabilities_statistics(layer: LayerPrimitive) -> tuple[int, int, int]:
    if not hasattr(layer, WEIGHTS_PRUNING_ATTR):
        raise ValueError("Layer does not have pruning probabilities attribute.")

    # Retrieve pruning probabilities
    prune_probs = getattr(layer, WEIGHTS_PRUNING_ATTR)
    base_probs = getattr(layer, WEIGHTS_BASE_ATTR)
    flip_probs = getattr(layer, WEIGHTS_FLIPPING_ATTR)

    probabilities = torch.stack([base_probs, flip_probs, prune_probs], dim=-1)
    pruned_count = torch.sum(torch.argmax(probabilities, dim=-1) == 2).item()
    flipped_count = torch.sum(torch.argmax(probabilities, dim=-1) == 1).item()
    base_count = torch.sum(torch.argmax(probabilities, dim=-1) == 0).item()

    return base_count, flipped_count, pruned_count

def get_layer_composite_pruning_statistics_probabilities(self: LayerComposite) -> tuple[int, int, int]:
    layers = get_layers_primitive(self)

    base_total = 0
    flipped_total = 0
    pruned_total = 0

    for layer in layers:
        base, flip, pruned = get_parameters_probabilities_statistics(layer)
        base_total += base
        flipped_total += flip
        pruned_total += pruned

    return  base_total, flipped_total, pruned_total

def get_layer_composite_pruning_statistics(self: LayerComposite) -> tuple[float, float]:
    layers = get_layers_primitive(self)
    total = 0
    remaining = 0
    for layer in layers:
        layer_total, layer_remaining = get_parameters_pruning_statistics(layer)
        total += layer_total
        remaining += layer_remaining

    return total, remaining

def get_remaining_parameters_loss_steep(self: LayerComposite) -> tuple[float, torch.Tensor]:
    layers: List[LayerPrimitive] = get_layers_primitive(self)
    total = 0
    sigmoids = torch.tensor(0, device=get_device(), dtype=torch.float)
    for layer in layers:
        layer_total, layer_sigmoid = get_parameters_pruning_sigmoid_steep(layer)
        total += layer_total
        sigmoids += layer_sigmoid

    return total, sigmoids

def get_remaining_parameters_loss(self: LayerComposite) -> tuple[float, torch.Tensor]:
    layers: List[LayerPrimitive] = get_layers_primitive(self)
    total = 0
    activations = torch.tensor(0, device=get_device(), dtype=torch.float)
    for layer in layers:
        layer_total, layer_sigmoid = get_parameters_pruning(layer)
        total += layer_total

    return total, activations

def get_parameters_total_count(self: LayerComposite) -> int:
    layers: List[LayerPrimitive] = get_layers_primitive(self)
    total = 0
    for layer in layers:
        layer_total = get_parameters_total(layer)
        total += layer_total

    return total

def get_remaining_parameters_loss_masks_importance(self: LayerComposite) -> tuple[float, torch.Tensor]:
    layers: List[LayerPrimitive] = get_layers_primitive(self)
    total = 0
    activations = torch.tensor(0, device=get_device(), dtype=torch.float)
    for layer in layers:
        layer_total, layer_remaining = get_parameters_pruning(layer)
        total += layer_total
        activations += layer_remaining

    return total, activations

def get_layers_primitive(self: LayerComposite) -> List[LayerPrimitive]:
    layers: List[LayerPrimitive] = []
    for layer in self.registered_layers:
        if isinstance(layer, LayerPrimitive):
            layers.append(layer)
        elif isinstance(layer, LayerComposite):
            layers.extend(get_layers_primitive(layer))

    return layers

class LayerLinearMaskImportance(LayerPrimitive):
    def __init__(self, configs_linear: ConfigsLayerLinear, configs_network: ConfigsNetworkMasksImportance):
        super().__init__()

        self.in_features = configs_linear.in_features
        self.out_features = configs_linear.out_features
        self.bias_enabled = configs_linear.bias_enabled

        self.mask_pruning_enabled = configs_network['mask_pruning_enabled']
        self.weights_training_enabled = configs_network['weights_training_enabled']

        setattr(self, WEIGHTS_ATTR, nn.Parameter(torch.Tensor(self.out_features, self.in_features)))
        setattr(self, WEIGHTS_PRUNING_ATTR, nn.Parameter(torch.Tensor(self.out_features, self.in_features)))

        if self.bias_enabled:
            setattr(self, BIAS_ATTR, nn.Parameter(torch.Tensor(self.out_features)))
            getattr(self, BIAS_ATTR).requires_grad = self.weights_training_enabled

        getattr(self, WEIGHTS_ATTR).requires_grad = self.weights_training_enabled
        getattr(self, WEIGHTS_PRUNING_ATTR).requires_grad = self.mask_pruning_enabled

        self.init_parameters()

    def get_bias_enabled(self):
        return self.bias_enabled

    def enable_weights_training(self):
        self.weights_training_enabled = True
        getattr(self, WEIGHTS_ATTR).requires_grad = True
        getattr(self, BIAS_ATTR).requires_grad = True

    def get_applied_weights(self) -> any:
        masked_weight = getattr(self, WEIGHTS_ATTR)
        if self.mask_pruning_enabled:
            mask_changes = MaskPruningFunction.apply(getattr(self, WEIGHTS_PRUNING_ATTR))
            masked_weight = masked_weight * mask_changes

        return masked_weight

    def init_parameters(self):
        init = configs_get_layers_initialization("fcn")
        init(getattr(self, WEIGHTS_ATTR))
        nn.init.uniform_(getattr(self, WEIGHTS_PRUNING_ATTR), a=get_flow_params_init(), b=get_flow_params_init() * 2.5)

        weights = getattr(self, WEIGHTS_ATTR)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(getattr(self, BIAS_ATTR), -bound, bound)

    def forward(self, input, inference = False):
        masked_weight = getattr(self, WEIGHTS_ATTR)
        bias = torch.zeros(self.out_features, device=get_device())
        if hasattr(self, BIAS_ATTR):
            bias = getattr(self, BIAS_ATTR)

        if self.mask_pruning_enabled:
            mask_changes = MaskPruningFunctionSigmoid.apply(getattr(self, WEIGHTS_PRUNING_ATTR))
            masked_weight = masked_weight * mask_changes

        return F.linear(input, masked_weight, bias)

@dataclass
class ConfigsLayerConv2:
    in_channels: int
    out_channels: int
    kernel_size: int
    padding: int = 0
    stride: int = 1
    bias_enabled: bool = True

class LayerConv2MaskImportance(LayerPrimitive):
    def __init__(self, configs_conv2d: ConfigsLayerConv2, configs_network_masks: ConfigsNetworkMasksImportance):
        super(LayerConv2MaskImportance, self).__init__()
        # getting configs
        self.in_channels = configs_conv2d.in_channels
        self.out_channels = configs_conv2d.out_channels
        self.kernel_size = configs_conv2d.kernel_size
        self.padding = configs_conv2d.padding
        self.stride = configs_conv2d.stride
        self.bias_enabled = configs_conv2d.bias_enabled

        self.mask_pruning_enabled = configs_network_masks['mask_pruning_enabled']
        self.weights_training_enabled = configs_network_masks['weights_training_enabled']

        # defining parameters
        setattr(self, WEIGHTS_ATTR, nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)))
        getattr(self, WEIGHTS_ATTR).requires_grad = self.weights_training_enabled

        setattr(self, WEIGHTS_PRUNING_ATTR, nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)))
        getattr(self, WEIGHTS_PRUNING_ATTR).requires_grad = self.mask_pruning_enabled

        if self.bias_enabled:
            setattr(self, BIAS_ATTR, nn.Parameter(torch.Tensor(self.out_channels)))
            getattr(self, BIAS_ATTR).requires_grad = self.weights_training_enabled

        self.init_parameters()

    def get_bias_enabled(self):
        return self.bias_enabled

    def get_applied_weights(self) -> any:
        masked_weights = getattr(self, WEIGHTS_ATTR)
        if self.mask_pruning_enabled:
            mask_changes = MaskPruningFunction.apply(getattr(self, WEIGHTS_PRUNING_ATTR))
            masked_weights = masked_weights * mask_changes

        return masked_weights

    def init_parameters(self):
        init = configs_get_layers_initialization("conv2d")
        init(getattr(self, WEIGHTS_ATTR))
        nn.init.uniform_(getattr(self, WEIGHTS_PRUNING_ATTR), a=get_flow_params_init(), b=get_flow_params_init() * 2.5)

        if hasattr(self, BIAS_ATTR):
            weights = getattr(self, WEIGHTS_ATTR)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(getattr(self, BIAS_ATTR), -bound, bound)

    def forward(self, input):
        masked_weights = getattr(self, WEIGHTS_ATTR)
        bias = torch.zeros(self.out_channels, device=get_device())
        if hasattr(self, BIAS_ATTR):
            bias = getattr(self, BIAS_ATTR)

        if self.mask_pruning_enabled:
            mask_changes = MaskPruningFunction.apply(getattr(self, WEIGHTS_PRUNING_ATTR))
            masked_weights = masked_weights * mask_changes

        return F.conv2d(input, masked_weights, bias, self.stride, self.padding)