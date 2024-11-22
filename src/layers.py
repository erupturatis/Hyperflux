from typing import TypedDict
from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import TypedDict
from src.config_layers import MaskPruningFunction, MaskFlipFunction, get_parameters_pruning, \
    get_parameters_pruning_statistics, get_parameters_flipped_statistics, configs_get_layers_initialization, \
    get_parameters_pruning_statistics_vanilla_network
from src.parameters_mask_processors import get_parameters_pruning_sigmoid_, get_parameters_pruning_statistics_sigmoid_, \
    get_parameters_flipped_statistics_sigmoid_, get_parameters_pruning_sigmoid_steep, get_parameters_total, \
    get_parameters_pruning_statistics_vanilla_
from src.mask_functions import MaskPruningFunctionSigmoid, MaskFlipFunctionSigmoid
from src.others import get_device
import math
import numpy as np
from src.constants import WEIGHTS_ATTR, BIAS_ATTR, WEIGHTS_PRUNING_ATTR, WEIGHTS_FLIPPING_ATTR

class ConfigsNetworkMasks(TypedDict):
    mask_pruning_enabled: bool
    weights_training_enabled: bool
    mask_flipping_enabled: bool


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

def get_layer_composite_flipped_statistics(self: LayerComposite) -> tuple[float, float]:
    layers = get_layers_primitive(self)
    total = 0
    remaining = 0
    for layer in layers:
        layer_total, layer_remaining = get_parameters_flipped_statistics(layer)
        total += layer_total
        remaining += layer_remaining

    return total, remaining


def get_layer_composite_pruning_statistics_vanilla(self: LayerComposite) -> tuple[float, float]:
    layers = get_layers_primitive(self)
    total = 0
    remaining = 0
    for layer in layers:
        layer_total, layer_remaining = get_parameters_pruning_statistics_vanilla_network(layer)
        total += layer_total
        remaining += layer_remaining

    return total, remaining

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

def get_remaining_parameters_loss(self: LayerComposite) -> tuple[float, torch.Tensor]:
    layers: List[LayerPrimitive] = get_layers_primitive(self)
    total = 0
    activations = torch.tensor(0, device=get_device(), dtype=torch.float)
    for layer in layers:
        layer_total, layer_sigmoid = get_parameters_pruning(layer)
        total += layer_total
        activations += layer_sigmoid

    return total, activations

def get_layers_primitive(self: LayerComposite) -> List[LayerPrimitive]:
    layers: List[LayerPrimitive] = []
    for layer in self.registered_layers:
        if isinstance(layer, LayerPrimitive):
            layers.append(layer)
        elif isinstance(layer, LayerComposite):
            layers.extend(get_layers_primitive(layer))

    return layers

class LayerLinear(LayerPrimitive):
    def __init__(self, configs_linear: ConfigsLayerLinear, configs_network: ConfigsNetworkMasks):
        super().__init__()

        self.in_features = configs_linear.in_features
        self.out_features = configs_linear.out_features
        self.bias_enabled = configs_linear.bias_enabled

        self.mask_pruning_enabled = configs_network['mask_pruning_enabled']
        self.weights_training_enabled = configs_network['weights_training_enabled']
        self.mask_flipping_enabled = configs_network['mask_flipping_enabled']


        setattr(self, WEIGHTS_ATTR, nn.Parameter(torch.Tensor(self.out_features, self.in_features)))
        setattr(self, WEIGHTS_PRUNING_ATTR, nn.Parameter(torch.Tensor(self.out_features, self.in_features)))
        setattr(self, WEIGHTS_FLIPPING_ATTR, nn.Parameter(torch.Tensor(self.out_features, self.in_features)))

        if self.bias_enabled:
            setattr(self, BIAS_ATTR, nn.Parameter(torch.Tensor(self.out_features)))
            getattr(self, BIAS_ATTR).requires_grad = self.weights_training_enabled

        getattr(self, WEIGHTS_ATTR).requires_grad = self.weights_training_enabled
        getattr(self, WEIGHTS_PRUNING_ATTR).requires_grad = self.mask_pruning_enabled
        getattr(self, WEIGHTS_FLIPPING_ATTR).requires_grad = self.mask_flipping_enabled

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

        if self.mask_flipping_enabled:
            mask_changes = MaskFlipFunction.apply(getattr(self, WEIGHTS_FLIPPING_ATTR))
            masked_weight = masked_weight * mask_changes

        return masked_weight

    def init_parameters(self):
        # nn.init.kaiming_uniform_(getattr(self, WEIGHTS_ATTR), a=math.sqrt(0))
        # nn.init.kaiming_uniform_(getattr(self, WEIGHTS_ATTR), a=math.sqrt(5))
        # nn.init.kaiming_normal_(getattr(self, WEIGHTS_ATTR), nonlinearity='relu')
        init = configs_get_layers_initialization("fcn")
        init(getattr(self, WEIGHTS_ATTR))
        
        nn.init.uniform_(getattr(self, WEIGHTS_PRUNING_ATTR), a=0.2, b=0.3)
        nn.init.uniform_(getattr(self, WEIGHTS_FLIPPING_ATTR), a=0.2, b=0.3)



        weights = getattr(self, WEIGHTS_ATTR)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(getattr(self, BIAS_ATTR), -bound, bound)

    def forward(self, input):
        masked_weight = getattr(self, WEIGHTS_ATTR)
        bias = torch.zeros(self.out_features, device=get_device())
        if hasattr(self, BIAS_ATTR):
            bias = getattr(self, BIAS_ATTR)

        if self.mask_pruning_enabled:
            mask_changes = MaskPruningFunction.apply(getattr(self, WEIGHTS_PRUNING_ATTR))
            masked_weight = masked_weight * mask_changes

        if self.mask_flipping_enabled:
            mask_changes = MaskPruningFunction.apply(getattr(self, WEIGHTS_FLIPPING_ATTR))
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

class LayerConv2(LayerPrimitive):
    def __init__(self, configs_conv2d: ConfigsLayerConv2, configs_network_masks: ConfigsNetworkMasks):
        super(LayerConv2, self).__init__()
        # getting configs
        self.in_channels = configs_conv2d.in_channels
        self.out_channels = configs_conv2d.out_channels
        self.kernel_size = configs_conv2d.kernel_size
        self.padding = configs_conv2d.padding
        self.stride = configs_conv2d.stride
        self.bias_enabled = configs_conv2d.bias_enabled

        self.mask_pruning_enabled = configs_network_masks['mask_pruning_enabled']
        self.mask_flipping_enabled = configs_network_masks['mask_flipping_enabled']
        self.weights_training_enabled = configs_network_masks['weights_training_enabled']

        # defining parameters
        setattr(self, WEIGHTS_ATTR, nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)))
        getattr(self, WEIGHTS_ATTR).requires_grad = self.weights_training_enabled

        setattr(self, WEIGHTS_PRUNING_ATTR, nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)))
        getattr(self, WEIGHTS_PRUNING_ATTR).requires_grad = self.mask_pruning_enabled

        setattr(self, WEIGHTS_FLIPPING_ATTR, nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)))
        getattr(self, WEIGHTS_FLIPPING_ATTR).requires_grad = self.mask_flipping_enabled

        if self.bias_enabled:
            setattr(self, BIAS_ATTR, nn.Parameter(torch.Tensor(self.out_channels)))
            getattr(self, BIAS_ATTR).requires_grad = self.weights_training_enabled

        # turning on and off params

        self.init_parameters()

    def get_bias_enabled(self):
        return self.bias_enabled

    def get_applied_weights(self) -> any:
        masked_weights = getattr(self, WEIGHTS_ATTR)
        if self.mask_pruning_enabled:
            mask_changes = MaskPruningFunction.apply(getattr(self, WEIGHTS_PRUNING_ATTR))
            masked_weights = masked_weights * mask_changes

        if self.mask_flipping_enabled:
            mask_changes = MaskFlipFunction.apply(getattr(self, WEIGHTS_FLIPPING_ATTR))
            masked_weights = masked_weights * mask_changes

        return masked_weights

    def init_parameters(self):
        init = configs_get_layers_initialization("conv2d")
        init(getattr(self, WEIGHTS_ATTR))

        nn.init.uniform_(getattr(self, WEIGHTS_PRUNING_ATTR), a=0.2, b=0.3)
        nn.init.uniform_(getattr(self, WEIGHTS_FLIPPING_ATTR), a=0.2, b=0.3)

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

        if self.mask_flipping_enabled:
            mask_changes = MaskFlipFunction.apply(getattr(self, WEIGHTS_FLIPPING_ATTR))
            masked_weights = masked_weights * mask_changes

        return F.conv2d(input, masked_weights, bias, self.stride, self.padding)



class LayerLinearVanilla(LayerPrimitive):
    def __init__(self, configs_linear: ConfigsLayerLinear):
        super().__init__()
        self.in_features = configs_linear.in_features
        self.out_features = configs_linear.out_features
        self.bias_enabled = configs_linear.bias_enabled

        # Define weight and bias parameters
        self.weights = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        if self.bias_enabled:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.init_parameters()

    def get_bias_enabled(self):
        return self.bias_enabled

    def get_applied_weights(self) -> torch.Tensor:
        return self.weights

    def init_parameters(self):
        # Initialize parameters using Kaiming Uniform
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias_enabled:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.get_applied_weights(), self.bias)


class LayerConv2Vanilla(LayerPrimitive):
    def __init__(self, configs_conv2d: ConfigsLayerConv2):
        super(LayerConv2Vanilla, self).__init__()
        self.in_channels = configs_conv2d.in_channels
        self.out_channels = configs_conv2d.out_channels
        self.kernel_size = configs_conv2d.kernel_size
        self.padding = configs_conv2d.padding
        self.stride = configs_conv2d.stride
        self.bias_enabled = configs_conv2d.bias_enabled

        # Define weight and bias parameters
        self.weights = nn.Parameter(
            torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        )
        if self.bias_enabled:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)

        self.init_parameters()

    def get_bias_enabled(self):
        return self.bias_enabled

    def get_applied_weights(self) -> torch.Tensor:
        return self.weights

    def init_parameters(self):
        # Initialize parameters using Kaiming Uniform
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias_enabled:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.conv2d(input, self.get_applied_weights(), self.bias, self.stride, self.padding)