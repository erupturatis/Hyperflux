from typing import TypedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import TypedDict

from src.mask_functions import MaskPruningFunction, MaskFlipFunction
from src.utils import get_device
import math
import numpy as np
from src.constants import WEIGHTS_ATTR, BIAS_ATTR, MASK_PRUNING_ATTR, MASK_FLIPPING_ATTR

class ConfigsNetworkMasks(TypedDict):
    mask_pruning_enabled: bool
    weights_training_enabled: bool
    mask_flipping_enabled: bool


class ConfigsLayerLinear(TypedDict):
    in_features: int
    out_features: int


class LayerLinear(nn.Module):
    def __init__(self, configs_linear: ConfigsLayerLinear, configs_network: ConfigsNetworkMasks):
        super(LayerLinear, self).__init__()

        self.in_features = configs_linear['in_features']
        self.out_features = configs_linear['out_features']

        self.mask_pruning_enabled = configs_network['mask_pruning_enabled']
        self.weights_training_enabled = configs_network['weights_training_enabled']
        self.mask_flipping_enabled = configs_network['mask_flipping_enabled']


        setattr(self, WEIGHTS_ATTR, nn.Parameter(torch.Tensor(self.out_features, self.in_features)))
        setattr(self, BIAS_ATTR, nn.Parameter(torch.Tensor(self.out_features)))

        getattr(self, WEIGHTS_ATTR).requires_grad = self.weights_training_enabled
        getattr(self, BIAS_ATTR).requires_grad = self.weights_training_enabled

        setattr(self, MASK_PRUNING_ATTR,  nn.Parameter(torch.Tensor(self.out_features, self.in_features)))
        setattr(self, MASK_FLIPPING_ATTR, nn.Parameter(torch.Tensor(self.out_features, self.in_features)))

        getattr(self, MASK_PRUNING_ATTR).requires_grad = self.mask_pruning_enabled
        getattr(self, MASK_FLIPPING_ATTR).requires_grad = self.mask_flipping_enabled

        self.init_parameters()

    def enable_weights_training(self):
        self.weights_training_enabled = True
        getattr(self, WEIGHTS_ATTR).requires_grad = True
        getattr(self, BIAS_ATTR).requires_grad = True

    def init_parameters(self):
        nn.init.kaiming_uniform_(getattr(self, WEIGHTS_ATTR), a=math.sqrt(5))
        nn.init.uniform_(getattr(self, MASK_PRUNING_ATTR), a=1, b=1)
        nn.init.uniform_(getattr(self, MASK_FLIPPING_ATTR), a=1, b=1)

        weights = getattr(self, WEIGHTS_ATTR)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(getattr(self, BIAS_ATTR), -bound, bound)

    def forward(self, input):
        masked_weight = getattr(self, WEIGHTS_ATTR)
        bias = getattr(self, BIAS_ATTR)

        if self.mask_pruning_enabled:
            mask_changes = MaskPruningFunction.apply(getattr(self, MASK_PRUNING_ATTR))
            masked_weight = masked_weight * mask_changes

        if self.mask_flipping_enabled:
            mask_changes = MaskFlipFunction.apply(self.mask_flipping)
            masked_weight = masked_weight * mask_changes

        return F.linear(input, masked_weight, bias)


class ConfigsLayerConv2(TypedDict):
    in_channels: int
    out_channels: int
    kernel_size: int
    padding: int
    stride: int

class LayerConv2(nn.Module):
    def __init__(self, configs_conv2d: ConfigsLayerConv2, configs_network_masks: ConfigsNetworkMasks):
        super(LayerConv2, self).__init__()
        # getting configs
        self.in_channels = configs_conv2d['in_channels']
        self.out_channels = configs_conv2d['out_channels']
        self.kernel_size = configs_conv2d['kernel_size']
        self.padding = configs_conv2d['padding']
        self.stride = configs_conv2d['stride']

        self.mask_pruning_enabled = configs_network_masks['mask_pruning_enabled']
        self.mask_flipping_enabled = configs_network_masks['mask_flipping_enabled']
        self.weights_training_enabled = configs_network_masks['weights_training_enabled']

        # defining parameters
        setattr(self, WEIGHTS_ATTR, nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)))
        setattr(self, MASK_PRUNING_ATTR, nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)))
        setattr(self, MASK_FLIPPING_ATTR, nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)))
        setattr(self, BIAS_ATTR, nn.Parameter(torch.Tensor(self.out_channels)))

        # turning on and off params
        getattr(self, WEIGHTS_ATTR).requires_grad = self.weights_training_enabled
        getattr(self, BIAS_ATTR).requires_grad = self.weights_training_enabled
        getattr(self, MASK_PRUNING_ATTR).requires_grad = self.mask_pruning_enabled
        getattr(self, MASK_FLIPPING_ATTR).requires_grad = self.mask_flipping_enabled

        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_uniform_(getattr(self, WEIGHTS_ATTR), a=math.sqrt(5))
        nn.init.uniform_(getattr(self, MASK_PRUNING_ATTR), a=1, b=1)
        nn.init.uniform_(getattr(self, MASK_FLIPPING_ATTR), a=0.5, b=0.5)

        weights = getattr(self, WEIGHTS_ATTR)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(getattr(self, BIAS_ATTR), -bound, bound)

    def forward(self, input):
        masked_weights = getattr(self, WEIGHTS_ATTR)
        bias = getattr(self, BIAS_ATTR)

        if self.mask_pruning_enabled:
            mask_changes = MaskPruningFunction.apply(getattr(self, MASK_PRUNING_ATTR))
            masked_weights = masked_weights * mask_changes

        if self.mask_flipping_enabled:
            mask_changes = MaskFlipFunction.apply(getattr(self, MASK_FLIPPING_ATTR))
            masked_weights = masked_weights * mask_changes

        return F.conv2d(input, masked_weights, bias, self.stride, self.padding)


class ConfigsBlockResnet(TypedDict):
    in_channels: int
    out_channels: int
    stride: int
    downsample: nn.Module
    mask_enabled: bool
    freeze_weights: bool
    signs_enabled: bool

class BlockResnet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, mask_enabled=True, freeze_weights=False, signs_enabled=True):
        super(BlockResnet, self).__init__()
        self.mask_enabled = mask_enabled
        self.signs_enabled = signs_enabled
        self.freeze_weights = freeze_weights

        self.conv1 = MaskedConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,mask_enabled=mask_enabled, freeze_weights=freeze_weights, signs_enabled=signs_enabled)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = MaskedConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, mask_enabled=mask_enabled, freeze_weights=freeze_weights, signs_enabled=signs_enabled)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample


    def get_masked_percentage_tensor(self) -> torch.Tensor:
        total = 0
        masked = torch.tensor(0, device=get_device(), dtype=torch.float)
        for layer in [self.conv1, self.conv2]:
            total += layer.weight.numel()
            mask = torch.sigmoid(layer.mask_param)
            masked += mask.sum()
        if self.downsample is not None:
            for sublayer in self.downsample:
                if hasattr(sublayer, 'mask_param'):
                    total += sublayer.weight.numel()
                    mask = torch.sigmoid(sublayer.mask_param)
                    masked += mask.sum()
        return masked, total

    def get_true_masked_percentage_tensor(self) -> torch.Tensor:
        total = 0
        masked = torch.tensor(0, device=get_device(), dtype=torch.float)

        for layer in [self.conv1, self.conv2]:
            total += layer.weight.numel()
            mask = torch.sigmoid(layer.mask_param)
            mask_thresholded = (mask >= 0.5).float()
            masked += mask_thresholded.sum()

        if self.downsample is not None:
            for sublayer in self.downsample:
                if hasattr(sublayer, 'mask_param'):
                    total += sublayer.weight.numel()
                    mask = torch.sigmoid(sublayer.mask_param)
                    mask_thresholded = (mask >= 0.5).float()
                    masked += mask_thresholded.sum()

        return masked, total

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
