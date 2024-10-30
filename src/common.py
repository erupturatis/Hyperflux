from typing import TypedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import TypedDict
from src.utils import get_device
import math
import numpy as np

from src.variables import WEIGHTS_ATTR, BIAS_ATTR, MASK_PRUNING_ATTR, MASK_FLIPPING_ATTR


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
        mask_changes = None

        if self.mask_pruning_enabled:
            mask_changes = MaskPruningFunction.apply(getattr(self, MASK_PRUNING_ATTR))
            masked_weight = masked_weight * mask_changes

        if self.mask_flipping_enabled:
            mask_changes = MaskFlipFunction.apply(self.mask_flipping)
            masked_weight = masked_weight * mask_changes

        return F.linear(input, masked_weight, bias)


