import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, mask_enabled=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mask_enabled = mask_enabled

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize mask parameters (unconstrained)
        self.mask_param = nn.Parameter(torch.Tensor(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.mask_param, a=-2, b=2)  # Initialize mask_param
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.mask_enabled:
            # Constrain mask values between 0 and 1 using sigmoid
            mask = torch.sigmoid(self.mask_param)
            # Apply threshold at 0.5
            mask_thresholded = (mask >= 0.5).float()
            # Apply mask to weights
            masked_weight = self.weight * mask_thresholded
        else:
            masked_weight = self.weight

        return F.linear(input, masked_weight, self.bias)

class Net(nn.Module):
    def __init__(self, mask_enabled=True):
        super(Net, self).__init__()
        self.fc1 = MaskedLinear(28*28, 300, mask_enabled=mask_enabled)
        self.fc2 = MaskedLinear(300, 100, mask_enabled=mask_enabled)
        self.fc3 = MaskedLinear(100, 10, mask_enabled=mask_enabled)

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
