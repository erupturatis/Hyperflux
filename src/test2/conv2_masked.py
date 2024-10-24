import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import get_device
import math
import numpy as np 

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
            self.bias.requires_grad = False


        self.mask_param = nn.Parameter(torch.Tensor(out_features, in_features))
        self.signs_mask_param = nn.Parameter(torch.Tensor(out_features, in_features))

        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #nn.init.uniform_(self.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.mask_param, a=1, b=1)  # Initialize mask_param, starts with all connections
        nn.init.uniform_(self.signs_mask_param, a=1, b=1)  # Initialize mask_param
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

class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding =0, stride = 1, bias=True, mask_enabled=True, freeze_weights=False, signs_enabled=True):
        super(MaskedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.mask_enabled = mask_enabled
        self.signs_enabled = signs_enabled
        
        self.weight =  nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)) 
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        if freeze_weights:
            self.weight.requires_grad = False
            self.bias.requires_grad = False

        self.mask_param = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)) 
        self.signs_mask_param = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)) 
    
        self.init_parameters()
        
    def init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a = 0, mode = 'fan_in')
        nn.init.uniform_(self.mask_param, a = 1, b = 1)
        nn.init.uniform_(self.signs_mask_param, a=1, b=1)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input):
        if self.mask_enabled:
            mask_thresholded = BinaryMaskFunction.apply(self.mask_param)
            masked_weight = self.weight * mask_thresholded
        else:
            masked_weight = self.weight
            
        if self.signs_enabled:
            mask_signs = SignMaskFunction.apply(self.signs_mask_param)
            masked_weight = masked_weight * mask_signs    
        else:
            masked_weight = masked_weight
        return F.conv2d(input, masked_weight, self.bias, self.stride, self.padding)


class Conv2(nn.Module):
    def __init__(self, mask_enabled = True, freeze_weights = False, signs_enabled = True):
        super(Conv2, self).__init__()

        self.conv2D_1 = MaskedConv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride = 1, bias = True, mask_enabled= mask_enabled, freeze_weights= freeze_weights, signs_enabled= signs_enabled)
        self.conv2D_2 = MaskedConv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride = 1, bias = True, mask_enabled= mask_enabled, freeze_weights= freeze_weights, signs_enabled= signs_enabled)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = MaskedLinear(64 * 16 * 16, 256, mask_enabled=mask_enabled, freeze_weights=freeze_weights, signs_enabled=signs_enabled)  
        self.fc2 = MaskedLinear(256, 256, mask_enabled=mask_enabled, freeze_weights=freeze_weights, signs_enabled=signs_enabled)
        self.fc3 = MaskedLinear(256, 10, mask_enabled=mask_enabled, freeze_weights=freeze_weights, signs_enabled=signs_enabled) 
        
        

        # nn.init.xavier_normal_(self.conv2D_1.weight)
        # nn.init.xavier_normal_(self.conv2D_2.weight)
        # nn.init.xavier_normal_(self.fc_1.weight)
        # nn.init.xavier_normal_(self.fc_2.weight)
        # nn.init.xavier_normal_(self.fc_3.weight)
        
        
    def get_masked_percentage_tensor(self) -> torch.Tensor:
        total = 0
        masked = torch.tensor(0, device=get_device(), dtype=torch.float)
        for layer in [self.fc1, self.fc2, self.fc3, self.conv2D_1, self.conv2D_2]:
            total += layer.weight.numel()
            mask = BinaryMaskFunction.apply(layer.mask_param)
            
            masked += mask.sum()

        return masked / total
    
    def forward(self, x):

        x = F.relu(self.conv2D_1(x))
        x = F.relu(self.conv2D_2(x))

        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x
