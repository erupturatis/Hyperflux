from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import TypedDict
from src.infrastructure.configs_layers import MaskPruningFunction, MaskFlipFunction, get_parameters_pruning, \
    get_parameters_pruning_statistics, get_parameters_flipped_statistics, configs_get_layers_initialization, \
    get_parameters_pruning_statistics_vanilla_network
from src.infrastructure.parameters_mask_processors import get_parameters_pruning_sigmoid_steep, get_parameters_total
from src.infrastructure.mask_functions import MaskPruningFunctionSigmoid, MaskPruningFunctionSigmoidDebugging
from src.infrastructure.others import get_device
import math
from src.infrastructure.constants import WEIGHTS_ATTR, BIAS_ATTR, WEIGHTS_PRUNING_ATTR, WEIGHTS_FLIPPING_ATTR, WEIGHTS_BASE_ATTR


class ConfigsNetworkMasksProbabilitiesPruneSign(TypedDict):
    mask_probabilities_enabled: bool
    weights_training_enabled: bool

class ConfigsNetworkMasksImportance(TypedDict):
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

class LayerLinearMaskImportance(LayerPrimitive):
    def __init__(self, configs_linear: ConfigsLayerLinear, configs_network: ConfigsNetworkMasksImportance):
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

        # nn.init.uniform_(getattr(self, WEIGHTS_PRUNING_ATTR), a=0.5, b=1)
        nn.init.uniform_(getattr(self, WEIGHTS_PRUNING_ATTR), a=0.3, b=0.5)
        nn.init.uniform_(getattr(self, WEIGHTS_FLIPPING_ATTR), a=0.2, b=0.3)

        weights = getattr(self, WEIGHTS_ATTR)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(getattr(self, BIAS_ATTR), -bound, bound)

    def debug_masks_values(self):
        MaskPruningFunctionSigmoidDebugging.apply(getattr(self, WEIGHTS_PRUNING_ATTR), getattr(self,WEIGHTS_ATTR))

    def forward(self, input, inference = False):
        masked_weight = getattr(self, WEIGHTS_ATTR)
        bias = torch.zeros(self.out_features, device=get_device())
        if hasattr(self, BIAS_ATTR):
            bias = getattr(self, BIAS_ATTR)

        if not hasattr(self, 'prev_pruning_mask'):
            self.prev_pruning_mask = torch.ones_like(getattr(self, WEIGHTS_PRUNING_ATTR), dtype=torch.int32)

        if self.mask_pruning_enabled:
            mask_changes = MaskPruningFunctionSigmoid.apply(getattr(self, WEIGHTS_PRUNING_ATTR))
            masked_weight = masked_weight * mask_changes

        if self.mask_flipping_enabled:
            mask_changes = MaskFlipFunction.apply(getattr(self, WEIGHTS_FLIPPING_ATTR))
            masked_weight = masked_weight * mask_changes

        return F.linear(input, masked_weight, bias)

    def get_and_reset_pruning_metrics(self):
        self.prev_pruning_mask = torch.ones_like(getattr(self, WEIGHTS_PRUNING_ATTR), dtype=torch.int32)
        print(self.pruning_metrics["flipped_state"])





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

        nn.init.uniform_(getattr(self, WEIGHTS_PRUNING_ATTR), a=0.2, b=0.5)
        nn.init.uniform_(getattr(self, WEIGHTS_FLIPPING_ATTR), a=0.2, b=0.5)

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



class LayerLinearMaskProbabilitiesPruneSign(LayerPrimitive):
    def __init__(self, configs_linear: ConfigsLayerLinear, configs_network: ConfigsNetworkMasksProbabilitiesPruneSign):
        super().__init__()

        self.in_features = configs_linear.in_features
        self.out_features = configs_linear.out_features

        self.mask_probabilities_enabled = configs_network['mask_probabilities_enabled']
        self.weights_training_enabled = configs_network['weights_training_enabled']

        setattr(self, WEIGHTS_ATTR, nn.Parameter(torch.Tensor(self.out_features, self.in_features)))
        setattr(self, BIAS_ATTR, nn.Parameter(torch.Tensor(self.out_features)))
        getattr(self, WEIGHTS_ATTR).requires_grad = self.weights_training_enabled
        getattr(self, BIAS_ATTR).requires_grad = self.weights_training_enabled

        # Attributes for base, flipped and pruned. They are passed in sigmoid
        setattr(self, WEIGHTS_BASE_ATTR, nn.Parameter(torch.Tensor(self.out_features, self.in_features)))
        setattr(self, WEIGHTS_PRUNING_ATTR, nn.Parameter(torch.Tensor(self.out_features, self.in_features)))
        setattr(self, WEIGHTS_FLIPPING_ATTR, nn.Parameter(torch.Tensor(self.out_features, self.in_features)))
        getattr(self, WEIGHTS_BASE_ATTR).requires_grad = self.mask_probabilities_enabled
        getattr(self, WEIGHTS_FLIPPING_ATTR).requires_grad = self.mask_probabilities_enabled
        getattr(self, WEIGHTS_PRUNING_ATTR).requires_grad = self.mask_probabilities_enabled

        self.init_parameters()

    def get_applied_weights(self) -> any:
        # To implement
        pass

    def init_parameters(self):
        init = configs_get_layers_initialization("fcn")
        init(getattr(self, WEIGHTS_ATTR))

        nn.init.uniform_(getattr(self, WEIGHTS_BASE_ATTR), a=1, b=2)
        nn.init.uniform_(getattr(self, WEIGHTS_PRUNING_ATTR), a=0.5, b=0.9)
        nn.init.uniform_(getattr(self, WEIGHTS_FLIPPING_ATTR), a=0.5, b=0.9)
        if self.mask_probabilities_enabled == False:
            nn.init.uniform_(getattr(self, WEIGHTS_PRUNING_ATTR), a=0, b=0)
            nn.init.uniform_(getattr(self, WEIGHTS_FLIPPING_ATTR), a=0, b=0)

        weights = getattr(self, WEIGHTS_ATTR)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(getattr(self, BIAS_ATTR), -bound, bound)

    def forward(self, input):
        weights = getattr(self, WEIGHTS_ATTR)
        bias = getattr(self, BIAS_ATTR)
        if self.mask_probabilities_enabled == False:
            return F.linear(input, weights, bias)

        base_probs = getattr(self, WEIGHTS_BASE_ATTR)
        flip_probs = getattr(self, WEIGHTS_FLIPPING_ATTR)
        prune_probs = getattr(self, WEIGHTS_PRUNING_ATTR)

        # Combine base, flip, and prune probabilities and normalize them
        probabilities = torch.softmax(torch.stack([base_probs, flip_probs, prune_probs], dim=-1), dim=-1)

        # Apply the straight-through mask
        mask = StraightThroughMask.apply(probabilities)

        # Modify weights based on the mask
        masked_weights = weights * mask

        return F.linear(input, masked_weights, bias)


    def forward_inference(self, input):
        weights = getattr(self, WEIGHTS_ATTR)
        bias = getattr(self, BIAS_ATTR)
        if not self.mask_probabilities_enabled:
            return F.linear(input, weights, bias)

        base_probs = getattr(self, WEIGHTS_BASE_ATTR)
        flip_probs = getattr(self, WEIGHTS_FLIPPING_ATTR)
        prune_probs = getattr(self, WEIGHTS_PRUNING_ATTR)

        # Combine base, flip, and prune probabilities and normalize them
        probabilities = torch.softmax(torch.stack([base_probs, flip_probs, prune_probs], dim=-1), dim=-1)

        # Select the action with the highest probability (0 = base, 1 = flip, 2 = prune)
        actions = torch.argmax(probabilities, dim=-1)

        # Apply the actions: 0 = base (x1), 1 = flip (x-1), 2 = prune (x0)
        mask = torch.where(actions == 0, 1.0, torch.where(actions == 1, -1.0, 0.0))

        # Modify weights based on the mask
        masked_weights = weights * mask

        return F.linear(input, masked_weights, bias)


class StraightThroughMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probabilities, temperature=1.0):
        """
        Forward pass for the Straight-Through Mask.

        Args:
            probabilities (Tensor): Probabilities for each mask state. Shape: (out_features, in_features, 3)
            temperature (float): Temperature parameter for Gumbel-Softmax.

        Returns:
            Tensor: Masked weights based on sampled actions. Shape: (out_features, in_features)
        """
        ctx.save_for_backward(probabilities)
        ctx.temperature = temperature

        # Sample from Gumbel-Softmax
        # Add small epsilon to avoid log(0)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(probabilities) + 1e-20) + 1e-20)
        y = (torch.log(probabilities + 1e-20) + gumbel_noise) / temperature
        soft_mask = F.softmax(y, dim=-1)

        # Hard sampling: one-hot vectors
        _, max_indices = soft_mask.max(dim=-1, keepdim=True)
        hard_mask = torch.zeros_like(probabilities).scatter_(-1, max_indices, 1.0)

        # Define mask values: 0 = base (1.0), 1 = flip (-1.0), 2 = prune (0.0)
        mask_values = torch.tensor([1.0, -1.0, 0.0], device=probabilities.device).view(1, 1, 3)
        hard_mask_values = torch.sum(hard_mask * mask_values, dim=-1)  # Shape: (out_features, in_features)

        # Straight-Through: Use hard mask in forward, soft mask for backward
        mask = hard_mask_values + soft_mask.sum(dim=-1) - soft_mask.detach().sum(dim=-1)

        return mask

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the Straight-Through Mask.

        Args:
            grad_output (Tensor): Gradient of the loss with respect to the output mask.

        Returns:
            Tensor: Gradient of the loss with respect to the input probabilities.
            None: No gradient for temperature.
        """
        probabilities, = ctx.saved_tensors
        temperature = ctx.temperature

        # Compute gradient through soft_mask
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(probabilities) + 1e-20) + 1e-20)
        y = (torch.log(probabilities + 1e-20) + gumbel_noise) / temperature
        soft_mask = F.softmax(y, dim=-1)

        # Gradient of the softmax
        grad_soft_mask = grad_output.unsqueeze(-1) * (soft_mask * (1 - soft_mask))

        return grad_soft_mask, None  # No gradient for temperature
