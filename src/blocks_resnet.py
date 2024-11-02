from typing import TypedDict
from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import TypedDict
from src.layers import LayerComposite, LayerPrimitive, ConfigsNetworkMasks, ConfigsLayerConv2, LayerConv2
from src.to_be_renamed import get_parameters_pruning_sigmoid, get_parameters_pruning_statistics, \
    get_parameters_flipped_statistics
from src.mask_functions import MaskPruningFunction, MaskFlipFunction
from src.others import get_device
import math
import numpy as np
from src.constants import WEIGHTS_ATTR, BIAS_ATTR, WEIGHTS_PRUNING_ATTR, WEIGHTS_FLIPPING_ATTR

# @dataclass
# class ConfigsBlockOfBlocks:
#     in_channels: int
#     out_channels: int
#     blocks: int
#     stride: int
#
# class BlockOfBlocksResnet(LayerComposite):
#     def __init__(self, configs_layer: ConfigsBlockOfBlocks, config_network_masks: ConfigsNetworkMasks):
#         super(BlockOfBlocksResnet, self).__init__()
#
#         self.EXPANSION = 1
#
#         stride = configs_layer["stride"]
#         in_channels = configs_layer["in_channels"]
#         out_channels = configs_layer["out_channels"]
#         blocks = configs_layer["blocks"]
#
#         if stride != 1 or in_channels != out_channels * self.EXPANSION:
#             downsample = True
#         else:
#             downsample = False
#
#         self.registered_layers = []
#         self.registered_layers.append(BlockResnet({
#             "in_channels": in_channels,
#             "out_channels": out_channels,
#             "stride": stride,
#             "padding": 1,
#             "kernel_size": 3,
#             "downsample": downsample
#         }, config_network_masks))
#
#         in_channels = out_channels * self.EXPANSION
#         for _ in range(1, blocks):
#             if stride != 1 or in_channels != out_channels * self.EXPANSION:
#                 downsample = True
#             else:
#                 downsample = False
#
#             self.registered_layers.append(BlockResnet({
#                 "in_channels": in_channels,
#                 "out_channels": out_channels,
#                 "stride": 1,
#                 "padding": 1,
#                 "kernel_size": 3,
#                 "downsample": downsample
#             }, config_network_masks))
#
#         self.registered_layers = nn.ParameterList(self.registered_layers)
#
#     def get_remaining_parameters_loss(self) -> tuple[float, torch.Tensor]:
#         return get_remaining_parameters_loss(self)
#
#     def get_layers_primitive(self) -> List[LayerPrimitive]:
#         return get_layers_primitive(self)
#
#     def get_parameters_prunning_statistics(self) -> any:
#         return get_layer_composite_pruning_statistics(self)
#
#     def forward(self, x):
#         for layer in self.registered_layers:
#             x = layer(x)
#
#         return x

@dataclass
class ConfigsBlockResnet:
    out_channels: int
    in_channels: int
    kernel_size: int
    stride: int
    padding: int
    downsample: bool


class BlockResnet(LayerComposite):
    def __init__(self, configs_block: ConfigsBlockResnet, configs_network_masks: ConfigsNetworkMasks):
        super(BlockResnet, self).__init__()

        in_channels = configs_block.in_channels
        out_channels = configs_block.out_channels
        stride = configs_block.stride
        padding = configs_block.padding
        kernel = configs_block.kernel_size
        downsample = configs_block.downsample

        self.conv1 = LayerConv2(ConfigsLayerConv2(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding), configs_network_masks)
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

