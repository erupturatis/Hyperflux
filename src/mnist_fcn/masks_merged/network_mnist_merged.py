import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.layers import ConfigsLayerLinear
from src.mnist_fcn.masks_merged.others import ConfigsNetworkMasksMerged

from src.others import get_device
from src.constants import WEIGHTS_ATTR, BIAS_ATTR, MASK_MERGED_ATTR

INTERVAL_SIZE = 1

def apply_forward(input_param):
    """
    Between [-inf, -interval] the mask is -1, between [-interval, interval] the mask is 0, and between [interval, inf] the mask is 1
    :param ctx:
    :param input_param:
    :return:
    """
    interval_size = INTERVAL_SIZE
    mask = input_param
    output = (mask > interval_size).float() - (mask < -interval_size).float()
    return output

class MaskMergedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_param):
        """
        Between [-inf, -interval] the mask is -1, between [-interval, interval] the mask is 0, and between [interval, inf] the mask is 1
        :param ctx:
        :param input_param:
        :return:
        """
        interval_size = INTERVAL_SIZE
        mask = input_param

        # onesi1 = (mask > interval_size).float()
        # onesi2 = (mask < -interval_size).float()
        # onesmiddle = torch.ones_like(mask) - onesi1 - onesi2

        # First one has 1's after [interval, inf] and zeros [0, interval]
        # Second one has 1's before [-inf, -interval] and zeros [-interval, 0]
        # Subtraction gives -1
        output = (mask > interval_size).float() - (mask < -interval_size).float()
        # output = torch.zeros_like(mask) + onesi1 - onesmiddle

        ctx.save_for_backward(mask)
        ctx.interval_size = interval_size  # Save threshold for backward
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Between [-inf, -interval*1.5] the derivative is flat, between [-interval*1.5, -interval*0.5] the derivative is steep,
        between [-interval*0.5, interval*0.5] the derivative is flat, between [interval*0.5, interval*1.5] the derivative is steep,
        and between [interval*1.5, inf] the derivative is flat
        :param ctx:
        :param grad_output:
        :return:
        """
        mask, = ctx.saved_tensors
        threshold = ctx.interval_size
        base_gradient_value = 0.2  # Base gradient value

        # Compute derivative based on the specified ranges
        # Has 3 plateau regions: [-inf, -1.5], [-0.5, 0.5], [1.5, inf]
        # Has 3 aggressive regions: [-1,5 -0.5], [0.5, 1.5]

        # The regions intervals in forward are [-1, 1] -> 0, [1, inf] -> 1, [-inf, -1] -> -1
        # All of these are scaled by interval_size

        abs_mask = mask.abs()
        t_half = threshold / 2
        t_three_half = threshold * 1.5

        # maps the steep regions to 1
        middle_interval = (abs_mask >= t_half) & (abs_mask <= t_three_half)

        # start with all plateau and then add the aggressive regions
        derivative = base_gradient_value * (1 + middle_interval.float())

        return grad_output * derivative


class LayerLinearMerged(nn.Module):
    def __init__(self, configs_linear: ConfigsLayerLinear, configs_network: ConfigsNetworkMasksMerged):
        super(LayerLinearMerged, self).__init__()

        self.in_features = configs_linear['in_features']
        self.out_features = configs_linear['out_features']

        self.weights_training_enabled = configs_network['weights_training_enabled']
        self.mask_merged_enabled = configs_network['mask_merged_enabled']

        setattr(self, WEIGHTS_ATTR, nn.Parameter(torch.Tensor(self.out_features, self.in_features)))
        setattr(self, BIAS_ATTR, nn.Parameter(torch.Tensor(self.out_features)))

        getattr(self, WEIGHTS_ATTR).requires_grad = self.weights_training_enabled
        getattr(self, BIAS_ATTR).requires_grad = self.weights_training_enabled

        setattr(self, MASK_MERGED_ATTR, nn.Parameter(torch.Tensor(self.out_features, self.in_features)))
        getattr(self, MASK_MERGED_ATTR).requires_grad = self.mask_merged_enabled

        self.init_parameters()

    def set_mask(self, value:bool):
        self.mask_merged_enabled = value
        getattr(self, MASK_MERGED_ATTR).requires_grad = self.mask_merged_enabled

    def disable_mask_grad(self):
        getattr(self, MASK_MERGED_ATTR).requires_grad = False

    def enable_weights_training(self):
        self.weights_training_enabled = True
        getattr(self, WEIGHTS_ATTR).requires_grad = True
        getattr(self, BIAS_ATTR).requires_grad = True


    def init_parameters(self):
        nn.init.kaiming_uniform_(getattr(self, WEIGHTS_ATTR), a=math.sqrt(5))
        # starts from flat derivative
        nn.init.uniform_(getattr(self, MASK_MERGED_ATTR), a=INTERVAL_SIZE * 2, b=INTERVAL_SIZE * 2)
        # nn.init.uniform_(getattr(self, MASK_MERGED_ATTR), a=-0.3, b=0.3)

        weights = getattr(self, WEIGHTS_ATTR)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(getattr(self, BIAS_ATTR), -bound, bound)

    def forward(self, input):
        masked_weight = getattr(self, WEIGHTS_ATTR)
        bias = getattr(self, BIAS_ATTR)

        if self.mask_merged_enabled:
            mask_changes = MaskMergedFunction.apply(getattr(self, MASK_MERGED_ATTR))
            masked_weight = masked_weight * mask_changes
        return F.linear(input, masked_weight, bias)

class ModelMnistFNNMergedMask(nn.Module):
    def __init__(self, config_network_mask: ConfigsNetworkMasksMerged):
        super(ModelMnistFNNMergedMask, self).__init__()
        self.fc1 = LayerLinearMerged(
            configs_linear={'in_features': 28*28, 'out_features': 300},
            configs_network=config_network_mask
        )
        self.fc2 = LayerLinearMerged(
            configs_linear={'in_features': 300, 'out_features': 100},
            configs_network=config_network_mask
        )
        self.fc3 = LayerLinearMerged(
            configs_linear={'in_features': 100, 'out_features': 10},
            configs_network=config_network_mask
        )

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def enable_weights_training(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            layer.enable_weights_training()

    def get_add_connections_regularization_loss(self) -> torch.Tensor:
        total = 0
        masked = torch.tensor(0, device=get_device(), dtype=torch.float)
        for layer in [self.fc1, self.fc2, self.fc3]:
            total += layer.weights.numel()
            mask = getattr(layer, MASK_MERGED_ATTR)

            # Thresholds maximum mask values between -1 and 1 (we dont really make a difference between 1.5 and 3, since they are both 1 anyway)
            mask = torch.tanh(mask)
            mask = mask.abs()

            mask = 1 - mask
            # Any non-zero values should converge to 0 (prune)
            masked += mask.sum()

        return masked / total

    def get_prune_regularization_loss(self) -> torch.Tensor:
        total = 0
        masked = torch.tensor(0, device=get_device(), dtype=torch.float)
        for layer in [self.fc1, self.fc2, self.fc3]:
            enabled = getattr(layer, MASK_MERGED_ATTR).requires_grad
            if not enabled:
                continue

            total += layer.weights.numel()
            mask = getattr(layer, MASK_MERGED_ATTR)

            # Thresholds maximum mask values between -1 and 1 (we dont really make a difference between 1.5 and 3, since they are both 1 anyway)
            mask = torch.tanh(mask)
            # Any non-zero values should converge to 0 (prune)
            mask = mask.abs()
            masked += mask.sum()

        return masked / total

    def get_masked_percentages_per_layer(self) -> any:
        total_arr = []

        count_minus_ones_array = []
        count_zeros_array = []
        count_ones_array = []

        for layer in [self.fc1, self.fc2, self.fc3]:
            total = layer.weights.numel()
            mask = apply_forward(getattr(layer, MASK_MERGED_ATTR))

            count_minus_ones = torch.tensor(0, device=get_device(), dtype=torch.float)
            count_zeros = torch.tensor(0, device=get_device(), dtype=torch.float)
            count_ones = torch.tensor(0, device=get_device(), dtype=torch.float)
            # Count occurrences of -1, 0, and 1
            count_minus_ones += (mask == -1).sum().float()
            count_zeros += (mask == 0).sum().float()
            count_ones += (mask == 1).sum().float()

            total_arr.append(total)
            count_minus_ones_array.append(count_minus_ones)
            count_zeros_array.append(count_zeros)
            count_ones_array.append(count_ones)

        # Calculate percentages
        percent_minus_ones_arr = [ (count_minus_ones / total).item() * 100 for count_minus_ones, total in zip(count_minus_ones_array, total_arr)]
        percent_zeros_arr = [ (count_zeros / total).item() * 100 for count_zeros, total in zip(count_zeros_array, total_arr)]
        percent_ones_arr = [ (count_ones / total).item() * 100 for count_ones, total in zip(count_ones_array, total_arr)]

        return percent_minus_ones_arr, percent_zeros_arr, percent_ones_arr

    def get_masked_percentages(self) -> tuple[float, float, float]:
        total = 0
        count_minus_ones = torch.tensor(0, device=get_device(), dtype=torch.float)
        count_zeros = torch.tensor(0, device=get_device(), dtype=torch.float)
        count_ones = torch.tensor(0, device=get_device(), dtype=torch.float)

        for layer in [self.fc1, self.fc2, self.fc3]:
            total += layer.weights.numel()
            mask = apply_forward(getattr(layer, MASK_MERGED_ATTR))

            # Count occurrences of -1, 0, and 1
            count_minus_ones += (mask == -1).sum().float()
            count_zeros += (mask == 0).sum().float()
            count_ones += (mask == 1).sum().float()

        # Calculate percentages
        percent_minus_ones = (count_minus_ones / total).item() * 100
        percent_zeros = (count_zeros / total).item() * 100
        percent_ones = (count_ones / total).item() * 100

        return percent_minus_ones, percent_zeros, percent_ones

    def prune10percent(self, percent:float):
        """
        Prune the lowest 10% of the remaining active weights by magnitude in each layer.
        """
        for layer in [self.fc1, self.fc2, self.fc3]:
            weights = getattr(layer, WEIGHTS_ATTR)
            mask_param = getattr(layer, MASK_MERGED_ATTR)

            # Flatten weights and mask parameters
            weights_flat = weights.view(-1)
            mask_param_flat = mask_param.view(-1)

            # Apply current mask to identify active weights
            with torch.no_grad():
                current_mask = apply_forward(mask_param)
                current_mask_flat = current_mask.view(-1)
                active_indices = (current_mask_flat != 0).nonzero(as_tuple=True)[0]

            num_active_weights = active_indices.numel()

            if num_active_weights == 0:
                continue  # No active weights to prune in this layer

            # Compute magnitudes of active weights
            active_weights_flat = weights_flat[active_indices]
            active_magnitudes = active_weights_flat.abs()

            # Determine number of weights to prune (10% of active weights)
            num_weights_to_prune = max(int(num_active_weights * percent), 1)  # At least 1 weight

            # Find the pruning threshold among active weights
            sorted_magnitudes, sorted_indices = torch.sort(active_magnitudes)
            threshold = sorted_magnitudes[num_weights_to_prune - 1]

            # Create mask for weights to prune among active weights
            prune_mask = active_magnitudes <= threshold

            # Get the indices of weights to prune in the original flat tensor
            prune_indices = active_indices[prune_mask]

            # Adjust mask parameters
            with torch.no_grad():
                # Set mask parameters to 0 for pruned weights
                mask_param_flat[prune_indices] = 0
                # Optionally, reinforce active mask parameters
                mask_param_flat[active_indices[~prune_mask]] = INTERVAL_SIZE * 2

            # Reshape mask parameters back to their original shape
            mask_param.data = mask_param_flat.view_as(mask_param)

