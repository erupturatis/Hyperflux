from typing import TypedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import TypedDict
from src.utils import get_device
import math
import numpy as np

from src.variables import WEIGHTS_ATTR, BIAS_ATTR, MASK_PRUNING_ATTR, MASK_FLIPPING_ATTR


class ConfigsConv2D(TypedDict):
    in_channels: int
    out_channels: int
    kernel_size: int
    padding: int
    stride: int

class ConfigsNetworkMasks(TypedDict):
    mask_pruning_enabled: bool
    weights_training_enabled: bool
    mask_flipping_enabled: bool


class ConfigsLinear(TypedDict):
    in_features: int
    out_features: int


class MaskFlipFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_param):
        mask = torch.sigmoid(mask_param)
        mask_classified = torch.where(mask < 0.5, -1,1)
        ctx.save_for_backward(mask)
        return mask_classified.float()

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        # sigmoid derivative
        grad_mask_param = grad_output * mask * (1 - mask)
        return grad_mask_param


class MaskPruningFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_param):
        mask = torch.sigmoid(mask_param)
        mask_thresholded = (mask >= 0.5).float()
        ctx.save_for_backward(mask)
        return mask_thresholded

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        # sigmoid derivative
        grad_mask_param = grad_output * mask * (1 - mask)
        return grad_mask_param


class LayerLinear(nn.Module):
    def __init__(self, configs_linear: ConfigsLinear, configs_network: ConfigsNetworkMasks):
        super(LayerLinear, self).__init__()
        print(configs_network)
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
        mask_changes = None

        if self.mask_pruning_enabled:
            mask_changes = MaskPruningFunction.apply(getattr(self, MASK_PRUNING_ATTR))
            masked_weight = masked_weight * mask_changes

        if self.mask_flipping_enabled:
            mask_changes = MaskFlipFunction.apply(self.mask_flipping)
            masked_weight = masked_weight * mask_changes

        return F.linear(input, masked_weight, bias)


class LayerConv2(nn.Module):
    def __init__(self, configs_conv2d: ConfigsConv2D, configs_network_masks: ConfigsNetworkMasks):
        super(LayerConv2, self).__init__()
        # getting configs
        print(configs_network_masks)
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
        mask_changes = None

        if self.mask_pruning_enabled:
            mask_changes = MaskPruningFunction.apply(getattr(self, MASK_PRUNING_ATTR))
            masked_weights = masked_weights * mask_changes

        if self.mask_flipping_enabled:
            mask_changes = MaskFlipFunction.apply(getattr(self, MASK_FLIPPING_ATTR))
            masked_weights = masked_weights * mask_changes

        return F.conv2d(input, masked_weights, bias, self.stride, self.padding)
