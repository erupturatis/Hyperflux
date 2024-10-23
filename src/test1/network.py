import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
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
        
    def get_layerwise_weights(self):
        weights = {}
        for i, layer in enumerate([self.fc1, self.fc2, self.fc3]):
            weight = layer.weight.detach().cpu().numpy()
            weights[f'fc{i}_weights'] = np.where(weight < 0, -1, 1)
        return weights                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    
    def get_layerwise_masks(self):
        masks = {}
        for i, layer in enumerate([self.fc1, self.fc2, self.fc3]):
            if layer.mask_enabled:
                binary_mask = BinaryMaskFunction.apply(layer.mask_param).detach().cpu().numpy()
                print(binary_mask)
            if layer.signs_enabled:
                masks[f'fc{i}_weights'] = SignMaskFunction.apply(layer.signs_mask_param).detach().cpu().numpy()
        return masks
      
    def get_masked_percentage_tensor(self) -> torch.Tensor:
        total = 0
        masked = torch.tensor(0, device=get_device(), dtype=torch.float)
        for layer in [self.fc1, self.fc2, self.fc3]:
            total += layer.weight.numel()
            mask = BinaryMaskFunction.apply(layer.mask_param)
            
            masked += mask.sum()

        return masked / total
    def save_weights(self):
        # Apply masks to weights before saving
        masked_fc1_weight = self.fc1.weight * BinaryMaskFunction.apply(self.fc1.mask_param)
        masked_fc2_weight = self.fc2.weight * BinaryMaskFunction.apply(self.fc2.mask_param)
        masked_fc3_weight = self.fc3.weight * BinaryMaskFunction.apply(self.fc3.mask_param)
        
        if self.fc1.signs_enabled:
            masked_fc1_weight = masked_fc1_weight * SignMaskFunction.apply(self.fc1.signs_mask_param)
        if self.fc2.signs_enabled:
            masked_fc2_weight = masked_fc2_weight * SignMaskFunction.apply(self.fc2.signs_mask_param)
        if self.fc3.signs_enabled:
            masked_fc3_weight = masked_fc3_weight * SignMaskFunction.apply(self.fc3.signs_mask_param)

        # Save the masked weights
        weights_to_save = {
            'fc1.weight': masked_fc1_weight.detach().cpu(),
            'fc2.weight': masked_fc2_weight.detach().cpu(),
            'fc3.weight': masked_fc3_weight.detach().cpu()
        }

        if self.fc1.bias is not None:
            weights_to_save['fc1.bias'] = self.fc1.bias.detach().cpu()
        if self.fc2.bias is not None:
            weights_to_save['fc2.bias'] = self.fc2.bias.detach().cpu()
        if self.fc3.bias is not None:
            weights_to_save['fc3.bias'] = self.fc3.bias.detach().cpu()

        torch.save(weights_to_save, r"XAI_paper\nn_weights\model_v1_with_mask.pth")
        print(f"Weights with masks applied saved!")


