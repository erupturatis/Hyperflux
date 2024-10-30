from typing import TypedDict
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.layers import LayerConv2, ConfigsNetworkMasks, BlockResnet, LayerLinear
from src.utils import get_device


class ModelResnet(nn.Module):
    def __init__(self, configs_network_masks: ConfigsNetworkMasks):
        super(ModelResnet, self).__init__()

        self.NUM_CLASSES=1000
        self.IN_CHANNELS = 64
        self.registered_layers = []

        self.basic_block = BlockResnet

        self.LAYERS_COUNT = 4
        self.layers_blocks_count = [2, 2, 2, 2]
        self.layers_blocks_out_channels = [64, 128, 256, 512]

        self.mask_pruning_enabled = configs_network_masks["mask_pruning_enabled"]
        self.mask_flipping_enabled = configs_network_masks["mask_flipping_enabled"]
        self.weights_training_enabled = configs_network_masks["weights_training_enabled"]

        self.conv1 = LayerConv2({
           "in_channels": 3,
           "out_channels": 64,
           "padding": 3,
           "stride": 2,
           "kernel_size": 7
        }, configs_network_masks)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for idx in range(self.LAYERS_COUNT):
            block_count = self.layers_blocks_count[idx]
            block_channels = self.layers_blocks_out_channels[idx]

            setattr(self, f"layer{idx+1}", self._make_layer({
                "out_channels": block_channels,
                "blocks": block_count,
                "stride": 2
            }, configs_network_masks))
            self.registered_layers.append(f"layer{idx+1}")

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = LayerLinear({
            "in_features": 512 * self.basic_block.expansion,
            "out_features": self.NUM_CLASSES
        }, configs_network_masks)

        # self.load_pretrained_weights()

    class ConfigsLayer(TypedDict):
        out_channels: int
        blocks: int
        stride: int

    def _make_layer(self, configs_layer: ConfigsLayer, config_network_masks: ConfigsNetworkMasks):
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

