from typing import TypedDict
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.blocks_resnet import BlockResnet, ConfigsBlockResnet
from src.layers import LayerConv2, ConfigsNetworkMasks, LayerLinear, LayerComposite, LayerPrimitive, \
    get_layers_primitive, get_remaining_parameters_loss, get_layer_composite_pruning_statistics, ConfigsLayerConv2, \
    ConfigsLayerLinear, get_layer_composite_flipped_statistics
from dataclasses import dataclass


@dataclass
class ConfigsModelBaseResnet18:
    num_classes: int

class ModelBaseResnet18(LayerComposite):
    def __init__(self, configs_model_base_resnet: ConfigsModelBaseResnet18, configs_network_masks: ConfigsNetworkMasks):
        super(ModelBaseResnet18, self).__init__()
        self.registered_layers = []
        self.relu = nn.ReLU(inplace=True)

        self.NUM_OUTPUT_CLASSES = configs_model_base_resnet.num_classes

        # Initial convolutional layer with bias=False
        self.conv1 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.conv1)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 1 (Conv2_x): 2 Residual Blocks
        # Block 1
        self.layer1_block1_conv1 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer1_block1_conv1)
        self.layer1_block1_bn1 = nn.BatchNorm2d(64)

        self.layer1_block1_conv2 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer1_block1_conv2)
        self.layer1_block1_bn2 = nn.BatchNorm2d(64)

        # Block 2
        self.layer1_block2_conv1 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer1_block2_conv1)
        self.layer1_block2_bn1 = nn.BatchNorm2d(64)

        self.layer1_block2_conv2 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer1_block2_conv2)
        self.layer1_block2_bn2 = nn.BatchNorm2d(64)

        # Layer 2 (Conv3_x): 2 Residual Blocks
        # Block 1 with downsampling
        self.layer2_block1_conv1 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer2_block1_conv1)
        self.layer2_block1_bn1 = nn.BatchNorm2d(128)

        self.layer2_block1_conv2 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer2_block1_conv2)
        self.layer2_block1_bn2 = nn.BatchNorm2d(128)

        # Downsampling for residual connection
        self.layer2_block1_downsample = LayerConv2(
            ConfigsLayerConv2(
                in_channels=64,
                out_channels=128,
                kernel_size=1,
                stride=2,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer2_block1_downsample)
        self.layer2_block1_downsample_bn = nn.BatchNorm2d(128)

        # Block 2
        self.layer2_block2_conv1 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer2_block2_conv1)
        self.layer2_block2_bn1 = nn.BatchNorm2d(128)

        self.layer2_block2_conv2 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer2_block2_conv2)
        self.layer2_block2_bn2 = nn.BatchNorm2d(128)

        # Layer 3 (Conv4_x): 2 Residual Blocks
        # Block 1 with downsampling
        self.layer3_block1_conv1 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer3_block1_conv1)
        self.layer3_block1_bn1 = nn.BatchNorm2d(256)

        self.layer3_block1_conv2 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer3_block1_conv2)
        self.layer3_block1_bn2 = nn.BatchNorm2d(256)

        # Downsampling for residual connection
        self.layer3_block1_downsample = LayerConv2(
            ConfigsLayerConv2(
                in_channels=128,
                out_channels=256,
                kernel_size=1,
                stride=2,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer3_block1_downsample)
        self.layer3_block1_downsample_bn = nn.BatchNorm2d(256)

        # Block 2
        self.layer3_block2_conv1 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer3_block2_conv1)
        self.layer3_block2_bn1 = nn.BatchNorm2d(256)

        self.layer3_block2_conv2 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer3_block2_conv2)
        self.layer3_block2_bn2 = nn.BatchNorm2d(256)

        # Layer 4 (Conv5_x): 2 Residual Blocks
        # Block 1 with downsampling
        self.layer4_block1_conv1 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer4_block1_conv1)
        self.layer4_block1_bn1 = nn.BatchNorm2d(512)

        self.layer4_block1_conv2 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer4_block1_conv2)
        self.layer4_block1_bn2 = nn.BatchNorm2d(512)

        # Downsampling for residual connection
        self.layer4_block1_downsample = LayerConv2(
            ConfigsLayerConv2(
                in_channels=256,
                out_channels=512,
                kernel_size=1,
                stride=2,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer4_block1_downsample)
        self.layer4_block1_downsample_bn = nn.BatchNorm2d(512)

        # Block 2
        self.layer4_block2_conv1 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer4_block2_conv1)
        self.layer4_block2_bn1 = nn.BatchNorm2d(512)

        self.layer4_block2_conv2 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer4_block2_conv2)
        self.layer4_block2_bn2 = nn.BatchNorm2d(512)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = LayerLinear(
            ConfigsLayerLinear(
                in_features=512,
                out_features=self.NUM_OUTPUT_CLASSES
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.fc)

    def get_remaining_parameters_loss(self) -> torch.Tensor:
        total, sigmoid =  get_remaining_parameters_loss(self)
        return sigmoid / total

    def get_layers_primitive(self) -> List[LayerPrimitive]:
        return get_layers_primitive(self)

    def get_parameters_pruning_statistics(self) -> any:
        return get_layer_composite_pruning_statistics(self)

    def get_parameters_flipped_statistics(self) -> any:
        return get_layer_composite_flipped_statistics(self)

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        # Layer 1
        # Block 1
        identity = x
        out = self.layer1_block1_conv1(x)
        out = self.layer1_block1_bn1(out)
        out = self.relu(out)
        out = self.layer1_block1_conv2(out)
        out = self.layer1_block1_bn2(out)
        out += identity
        out = self.relu(out)

        # Block 2
        identity = out
        out = self.layer1_block2_conv1(out)
        out = self.layer1_block2_bn1(out)
        out = self.relu(out)
        out = self.layer1_block2_conv2(out)
        out = self.layer1_block2_bn2(out)
        out += identity
        out = self.relu(out)

        # Layer 2
        # Block 1 with downsampling
        identity = out
        out = self.layer2_block1_conv1(out)
        out = self.layer2_block1_bn1(out)
        out = self.relu(out)
        out = self.layer2_block1_conv2(out)
        out = self.layer2_block1_bn2(out)

        identity = self.layer2_block1_downsample(identity)
        identity = self.layer2_block1_downsample_bn(identity)

        out += identity
        out = self.relu(out)

        # Block 2
        identity = out
        out = self.layer2_block2_conv1(out)
        out = self.layer2_block2_bn1(out)
        out = self.relu(out)
        out = self.layer2_block2_conv2(out)
        out = self.layer2_block2_bn2(out)
        out += identity
        out = self.relu(out)

        # Layer 3
        # Block 1 with downsampling
        identity = out
        out = self.layer3_block1_conv1(out)
        out = self.layer3_block1_bn1(out)
        out = self.relu(out)
        out = self.layer3_block1_conv2(out)
        out = self.layer3_block1_bn2(out)

        identity = self.layer3_block1_downsample(identity)
        identity = self.layer3_block1_downsample_bn(identity)

        out += identity
        out = self.relu(out)

        # Block 2
        identity = out
        out = self.layer3_block2_conv1(out)
        out = self.layer3_block2_bn1(out)
        out = self.relu(out)
        out = self.layer3_block2_conv2(out)
        out = self.layer3_block2_bn2(out)
        out += identity
        out = self.relu(out)

        # Layer 4
        # Block 1 with downsampling
        identity = out
        out = self.layer4_block1_conv1(out)
        out = self.layer4_block1_bn1(out)
        out = self.relu(out)
        out = self.layer4_block1_conv2(out)
        out = self.layer4_block1_bn2(out)

        identity = self.layer4_block1_downsample(identity)
        identity = self.layer4_block1_downsample_bn(identity)

        out += identity
        out = self.relu(out)

        # Block 2
        identity = out
        out = self.layer4_block2_conv1(out)
        out = self.layer4_block2_bn1(out)
        out = self.relu(out)
        out = self.layer4_block2_conv2(out)
        out = self.layer4_block2_bn2(out)
        out += identity
        out = self.relu(out)

        # Final layers
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    # def load_pretrained_weights(self, weight_path=r"C:\Users\Statia 1\Desktop\AlexoaieAntonio\XAI_paper\nn_weights\resnet18-f37072fd.pth"):
    #     """PLEASE REFACTOR THIS"""
    #     # Load weights from the specified .pth file
    #     pretrained_state = torch.load(weight_path)  # Load the state dictionary from file
    #
    #     # Get the current state dictionary of our model
    #     own_state = self.state_dict()
    #
    #     for name, param in pretrained_state.items():
    #         if name not in own_state:
    #             print(f"Parameter '{name}' does not match any layer in the model's own_state.")
    #             continue  # Ignore parameters that don’t match our model structure
    #
    #         if isinstance(param, nn.Parameter):
    #             param = param.data  # Convert parameters to tensors
    #
    #         # Check for size compatibility
    #         if own_state[name].size() != param.size():
    #             print(f"Skipping '{name}' due to size mismatch: expected {own_state[name].size()}, got {param.size()}.")
    #             continue
    #
    #         # Only load weights for the main layers (ignore mask/sign layers)
    #         if 'mask_param' not in name and 'signs_mask_param' not in name:
    #             own_state[name].copy_(param)
    #
    #     print(f"Loaded pretrained weights from {weight_path}.")
