import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.layers import LayerConv2
from src.utils import get_device
import torchvision.models as models

class ModelResnet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, mask_enabled=True, freeze_weights=False, signs_enabled=False):
        super(ModelResnet, self).__init__()
        self.mask_enabled = mask_enabled
        self.signs_enabled = signs_enabled
        self.freeze_weights = freeze_weights
        self.in_channels = 64

        self.conv1 = MaskedConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, mask_enabled=self.mask_enabled, freeze_weights=self.freeze_weights, signs_enabled=self.signs_enabled)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual block layers
        self.layer1 = self._make_layer(block, 64, layers[0], mask_enabled = self.mask_enabled, freeze_weights = self.freeze_weights, signs_enabled = self.signs_enabled)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, mask_enabled = self.mask_enabled, freeze_weights = self.freeze_weights, signs_enabled = self.signs_enabled)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, mask_enabled = self.mask_enabled, freeze_weights = self.freeze_weights, signs_enabled = self.signs_enabled)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, mask_enabled = self.mask_enabled, freeze_weights = self.freeze_weights, signs_enabled = self.signs_enabled)

      
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = MaskedLinear(512 * block.expansion, num_classes, mask_enabled=self.mask_enabled, freeze_weights=self.freeze_weights, signs_enabled=self.signs_enabled)
        self.load_pretrained_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1, mask_enabled=True, freeze_weights=False, signs_enabled=True):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                MaskedConv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False, mask_enabled=mask_enabled, freeze_weights=freeze_weights, signs_enabled=signs_enabled),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def get_masked_percentage_tensor(self) -> torch.Tensor:
        total = 0
        masked = torch.tensor(0, device=get_device(), dtype=torch.float)

        for layer in [self.fc, self.conv1]:
            total += layer.weight.numel()
            mask = torch.sigmoid(layer.mask_param)
            masked += mask.sum()

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                if hasattr(block, 'mask_param'):
                    aux_masked, aux_total = block.get_masked_percentage_tensor()
                    masked += aux_masked
                    total += aux_total
            
        return masked / total

    def get_true_masked_percentage_tensor(self) -> torch.Tensor:
        total = 0
        masked = torch.tensor(0, device=get_device(), dtype=torch.float)
        for layer in [self.fc, self.conv1]:
            total += layer.weight.numel()
            mask = torch.sigmoid(layer.mask_param)
            mask_thresholded = (mask >= 0.5).float()
            masked += mask_thresholded.sum()

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                if hasattr(block, 'mask_param'):
                    aux_masked, aux_total = block.get_true_masked_percentage_tensor()
                    masked += aux_masked
                    total += aux_total

        return masked / total
    def load_pretrained_weights(self, weight_path=r"C:\Users\Statia 1\Desktop\AlexoaieAntonio\XAI_paper\nn_weights\resnet18-f37072fd.pth"):
        # Load weights from the specified .pth file
        pretrained_state = torch.load(weight_path)  # Load the state dictionary from file

        # Get the current state dictionary of our model
        own_state = self.state_dict()

        for name, param in pretrained_state.items():
            if name not in own_state:
                print(f"Parameter '{name}' does not match any layer in the model's own_state.")
                continue  # Ignore parameters that don’t match our model structure

            if isinstance(param, nn.Parameter):
                param = param.data  # Convert parameters to tensors
            
            # Check for size compatibility
            if own_state[name].size() != param.size():
                print(f"Skipping '{name}' due to size mismatch: expected {own_state[name].size()}, got {param.size()}.")
                continue

            # Only load weights for the main layers (ignore mask/sign layers)
            if 'mask_param' not in name and 'signs_mask_param' not in name:
                own_state[name].copy_(param)

        print(f"Loaded pretrained weights from {weight_path}.")
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

