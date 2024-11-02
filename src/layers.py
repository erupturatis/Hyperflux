from typing import TypedDict
from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import TypedDict
from src.to_be_renamed import get_parameters_pruning_sigmoid, get_parameters_pruning_statistics, \
    get_parameters_flipped_statistics
from src.mask_functions import MaskPruningFunction, MaskFlipFunction
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
    bias: bool = True

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


def get_layer_composite_pruning_statistics(self: LayerComposite) -> tuple[float, float]:
    layers = get_layers_primitive(self)
    total = 0
    remaining = 0
    for layer in layers:
        layer_total, layer_remaining = get_parameters_pruning_statistics(layer)
        total += layer_total
        remaining += layer_remaining

    return total, remaining

def get_remaining_parameters_loss(self: LayerComposite) -> tuple[float, torch.Tensor]:
    layers: List[LayerPrimitive] = get_layers_primitive(self)
    total = 0
    sigmoids = torch.tensor(0, device=get_device(), dtype=torch.float)
    for layer in layers:
        layer_total, layer_sigmoid = get_parameters_pruning_sigmoid(layer)
        total += layer_total
        sigmoids += layer_sigmoid

    return total, sigmoids

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
        self.bias_enabled = configs_linear.bias

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

    def enable_weights_training(self):
        self.weights_training_enabled = True
        getattr(self, WEIGHTS_ATTR).requires_grad = True
        getattr(self, BIAS_ATTR).requires_grad = True

    def init_parameters(self):
        nn.init.kaiming_uniform_(getattr(self, WEIGHTS_ATTR), a=math.sqrt(5))
        nn.init.uniform_(getattr(self, WEIGHTS_PRUNING_ATTR), a=1, b=1)
        nn.init.uniform_(getattr(self, WEIGHTS_FLIPPING_ATTR), a=1, b=1)

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
            mask_changes = MaskFlipFunction.apply(self.mask_flipping)
            masked_weight = masked_weight * mask_changes

        return F.linear(input, masked_weight, bias)


@dataclass
class ConfigsLayerConv2:
    in_channels: int
    out_channels: int
    kernel_size: int
    padding: int = 0
    stride: int = 1
    bias: bool = True

class LayerConv2(LayerPrimitive):
    def __init__(self, configs_conv2d: ConfigsLayerConv2, configs_network_masks: ConfigsNetworkMasks):
        super(LayerConv2, self).__init__()
        # getting configs
        self.in_channels = configs_conv2d.in_channels
        self.out_channels = configs_conv2d.out_channels
        self.kernel_size = configs_conv2d.kernel_size
        self.padding = configs_conv2d.padding
        self.stride = configs_conv2d.stride
        self.bias_enabled = configs_conv2d.bias

        self.mask_pruning_enabled = configs_network_masks['mask_pruning_enabled']
        self.mask_flipping_enabled = configs_network_masks['mask_flipping_enabled']
        self.weights_training_enabled = configs_network_masks['weights_training_enabled']

        # defining parameters
        setattr(self, WEIGHTS_ATTR, nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)))
        setattr(self, WEIGHTS_PRUNING_ATTR, nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)))
        setattr(self, WEIGHTS_FLIPPING_ATTR, nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)))
        if self.bias_enabled:
            setattr(self, BIAS_ATTR, nn.Parameter(torch.Tensor(self.out_channels)))
            getattr(self, BIAS_ATTR).requires_grad = self.weights_training_enabled

        # turning on and off params
        getattr(self, WEIGHTS_ATTR).requires_grad = self.weights_training_enabled
        getattr(self, BIAS_ATTR).requires_grad = self.weights_training_enabled
        getattr(self, WEIGHTS_PRUNING_ATTR).requires_grad = self.mask_pruning_enabled
        getattr(self, WEIGHTS_FLIPPING_ATTR).requires_grad = self.mask_flipping_enabled

        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_uniform_(getattr(self, WEIGHTS_ATTR), a=math.sqrt(5))
        nn.init.uniform_(getattr(self, WEIGHTS_PRUNING_ATTR), a=1, b=1)
        nn.init.uniform_(getattr(self, WEIGHTS_FLIPPING_ATTR), a=0.5, b=0.5)

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


class ConfigsBlockDownsample(TypedDict):
    in_channels: int
    out_channels: int
    stride: int


class BlockDownsample(LayerComposite):
    def __init__(self, configs_block: ConfigsBlockDownsample, configs_network_masks: ConfigsNetworkMasks):
        super(BlockDownsample, self).__init__()

        self.EXPANSION = 1
        self.in_channels = configs_block['in_channels']
        self.out_channels = configs_block['out_channels']
        self.stride = configs_block['stride']

        self.conv1 = LayerConv2({
            "in_channels": self.in_channels,
            "out_channels": self.out_channels * self.EXPANSION,
            "kernel_size": 1,
            "stride": self.stride,
            "padding": 0
        }, configs_network_masks)
        self.bn = nn.BatchNorm2d(self.out_channels * self.EXPANSION)

        self.registered_layers = [self.conv1]


    def get_remaining_parameters_loss(self) -> tuple[float, torch.Tensor]:
        return get_remaining_parameters_loss(self)

    def get_layers_primitive(self) -> List[LayerPrimitive]:
        return get_layers_primitive(self)

    def get_parameters_prunning_statistics(self) -> any:
        return get_layer_composite_pruning_statistics(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)

        return x

class ConfigsBlockResnet(TypedDict):
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    downsample: bool

class ConfigsBlockOfBlocks(TypedDict):
    in_channels: int
    out_channels: int
    blocks: int
    stride: int

class BlockOfBlocksResnet(LayerComposite):
    def __init__(self, configs_layer: ConfigsBlockOfBlocks, config_network_masks: ConfigsNetworkMasks):
        super(BlockOfBlocksResnet, self).__init__()

        self.EXPANSION = 1

        stride = configs_layer["stride"]
        in_channels = configs_layer["in_channels"]
        out_channels = configs_layer["out_channels"]
        blocks = configs_layer["blocks"]

        if stride != 1 or in_channels != out_channels * self.EXPANSION:
            downsample = True
        else:
            downsample = False

        self.registered_layers = []
        self.registered_layers.append(BlockResnet({
            "in_channels": in_channels,
            "out_channels": out_channels,
            "stride": stride,
            "padding": 1,
            "kernel_size": 3,
            "downsample": downsample
        }, config_network_masks))

        in_channels = out_channels * self.EXPANSION
        for _ in range(1, blocks):
            if stride != 1 or in_channels != out_channels * self.EXPANSION:
                downsample = True
            else:
                downsample = False

            self.registered_layers.append(BlockResnet({
                "in_channels": in_channels,
                "out_channels": out_channels,
                "stride": 1,
                "padding": 1,
                "kernel_size": 3,
                "downsample": downsample
            }, config_network_masks))

        self.registered_layers = nn.ParameterList(self.registered_layers)

    def get_remaining_parameters_loss(self) -> tuple[float, torch.Tensor]:
        return get_remaining_parameters_loss(self)

    def get_layers_primitive(self) -> List[LayerPrimitive]:
        return get_layers_primitive(self)

    def get_parameters_prunning_statistics(self) -> any:
        return get_layer_composite_pruning_statistics(self)

    def forward(self, x):
        for layer in self.registered_layers:
            x = layer(x)

        return x

class BlockResnet(LayerComposite):
    def __init__(self, configs_block: ConfigsBlockResnet, configs_network_masks: ConfigsNetworkMasks):
        super(BlockResnet, self).__init__()

        self.mask_pruning_enabled = configs_network_masks['mask_pruning_enabled']
        self.mask_flipping_enabled = configs_network_masks['mask_flipping_enabled']
        self.weights_training_enabled = configs_network_masks['weights_training_enabled']

        in_channels = configs_block['in_channels']
        out_channels = configs_block['out_channels']
        stride = configs_block['stride']
        padding = configs_block['padding']
        kernel = configs_block['kernel_size']

        self.conv1 = LayerConv2({
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel,
            'padding': padding,
            'stride': stride
        }, configs_network_masks)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = LayerConv2({
            'in_channels': out_channels,
            'out_channels': out_channels,
            'kernel_size': kernel,
            'padding': padding,
            'stride': 1
        }, configs_network_masks)
        self.bn2 = nn.BatchNorm2d(out_channels)


        self.relu = nn.ReLU()

        self.registered_layers = [self.conv1, self.conv2]

        downsample = configs_block['downsample']
        if downsample:
            self.downsample = BlockDownsample({
                "in_channels": in_channels,
                "out_channels": out_channels,
                "stride": stride
            }, configs_network_masks)
            self.registered_layers.append(self.downsample)

        self.registered_layers = nn.ParameterList(self.registered_layers)

    def get_remaining_parameters_loss(self) -> tuple[float, torch.Tensor]:
        return get_remaining_parameters_loss(self)

    def get_layers_primitive(self) -> List[LayerPrimitive]:
        return get_layers_primitive(self)

    def get_parameters_prunning_statistics(self) -> any:
        return get_layer_composite_pruning_statistics(self)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if hasattr(self, 'downsample'):
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
