import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.utils import get_device

class SignMaskFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_param):
        # Apply sigmoid to constrain values between 0 and 1
        mask = torch.sigmoid(mask_param)
        # Apply conditions to return -1, 0, or 1
        mask_classified = torch.where(mask < 0.5, -1,1)
        # Save the sigmoid mask for use in backward pass
        ctx.save_for_backward(mask)
        return mask_classified.float()

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        # Approximate gradient: derivative of the sigmoid function
        grad_mask_param = grad_output * mask * (1 - mask)
        return grad_mask_param

class BinaryMaskFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_param):
        # Apply sigmoid to constrain values between 0 and 1
        mask = torch.sigmoid(mask_param)
        # Apply threshold at 0.5 to get binary mask
        mask_thresholded = (mask >= 0.5).float()
        # Save the sigmoid mask for use in backward pass
        ctx.save_for_backward(mask)
        return mask_thresholded

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        # Approximate gradient: derivative of the sigmoid function
        grad_mask_param = grad_output * mask * (1 - mask)
        return grad_mask_param

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, mask_enabled=True, freeze_weights=False, signs_enabled=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mask_enabled = mask_enabled
        self.signs_enabled = signs_enabled

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        if freeze_weights:
            self.weight.requires_grad = False
            # self.bias.requires_grad = False

        # Initialize mask parameters (unconstrained)
        self.mask_param = nn.Parameter(torch.Tensor(out_features, in_features))
        self.signs_mask_param = nn.Parameter(torch.Tensor(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.mask_param, a=2, b=2)  # Initialize mask_param, starts with all connections
        nn.init.uniform_(self.signs_mask_param, a=2, b=2)  # Initialize mask_param
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.mask_enabled:
            # Constrain mask values between 0 and 1 using sigmoid
            mask_thresholded = BinaryMaskFunction.apply(self.mask_param)

            # Apply mask to weights
            masked_weight = self.weight * mask_thresholded
        else:
            masked_weight = self.weight

        if self.signs_enabled:
            # Constrain mask values between -1 and 1 using sigmoid
            mask_signs = SignMaskFunction.apply(self.signs_mask_param)
            masked_weight = masked_weight * mask_signs
        else:
            masked_weight = masked_weight


        return F.linear(input, masked_weight, self.bias)

    def get_masked_percentage_tensor(self) -> torch.Tensor:
        mask = BinaryMaskFunction.apply(self.mask_param)
        return mask.sum() / mask.numel()

class Net(nn.Module):
    def __init__(self, mask_enabled=True, freeze_weights=False, signs_enabled=True):
        super(Net, self).__init__()
        self.fc1 = MaskedLinear(28*28, 300, mask_enabled=mask_enabled, freeze_weights=freeze_weights, signs_enabled=signs_enabled)
        self.fc2 = MaskedLinear(300, 100, mask_enabled=mask_enabled, freeze_weights=freeze_weights, signs_enabled=signs_enabled)
        self.fc3 = MaskedLinear(100, 10, mask_enabled=mask_enabled, freeze_weights=freeze_weights, signs_enabled=signs_enabled)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def set_mask_enabled(self, enabled):
        self.fc1.mask_enabled = enabled
        self.fc2.mask_enabled = enabled
        self.fc3.mask_enabled = enabled

    def get_masked_percentage_tensor(self) -> torch.Tensor:
        total = 0
        masked = torch.tensor(0, device=get_device(), dtype=torch.float)
        for layer in [self.fc1, self.fc2, self.fc3]:
            total += layer.weight.numel()
            mask = BinaryMaskFunction.apply(layer.mask_param)
            masked += mask.sum()

        return masked / total

