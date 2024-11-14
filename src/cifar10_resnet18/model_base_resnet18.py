from typing import TypedDict
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.cifar10_resnet18.model_resnet18_attributes import RESNET18_CIFAR10_REGISTERED_LAYERS_ATTRIBUTES, \
    RESNET18_CIFAR10_UNREGISTERED_LAYERS_ATTRIBUTES
from src.others import get_device, prefix_path_with_root
from src.blocks_resnet import BlockResnet, ConfigsBlockResnet
from src.layers import LayerConv2, ConfigsNetworkMasks, LayerLinear, LayerComposite, LayerPrimitive, \
    get_layers_primitive, get_remaining_parameters_loss, get_layer_composite_pruning_statistics, ConfigsLayerConv2, \
    ConfigsLayerLinear, get_layer_composite_flipped_statistics, get_parameters_total_count
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

        for layer_attr in RESNET18_CIFAR10_REGISTERED_LAYERS_ATTRIBUTES:
            name = layer_attr['name']
            type_ = layer_attr['type']

            if type_ == 'LayerConv2':
                layer = LayerConv2(
                    ConfigsLayerConv2(
                        in_channels=layer_attr['in_channels'],
                        out_channels=layer_attr['out_channels'],
                        kernel_size=layer_attr['kernel_size'],
                        stride=layer_attr['stride'],
                        padding=layer_attr['padding'],
                        bias_enabled=layer_attr['bias_enabled']
                    ),
                    configs_network_masks
                )
            elif type_ == 'LayerLinear':
                layer = LayerLinear(
                    ConfigsLayerLinear(
                        in_features=layer_attr['in_features'],
                        out_features=layer_attr['out_features']
                    ),
                    configs_network_masks
                )
            else:
                raise ValueError(f"Unsupported registered layer type: {type_}")

            setattr(self, name, layer)
            self.registered_layers.append(layer)

        for layer_attr in RESNET18_CIFAR10_UNREGISTERED_LAYERS_ATTRIBUTES:
            name = layer_attr['name']
            type_ = layer_attr['type']

            if type_ == 'BatchNorm2d':
                layer = nn.BatchNorm2d(
                    num_features=layer_attr['num_features']
                )
            elif type_ == 'AdaptiveAvgPool2d':
                layer = nn.AdaptiveAvgPool2d(
                    output_size=layer_attr['output_size']
                )
            else:
                raise ValueError(f"Unsupported unregistered layer type: {type_}")

            setattr(self, name, layer)

        self.load_pretrained_weights()

    def get_remaining_parameters_loss(self) -> torch.Tensor:
        total, sigmoid =  get_remaining_parameters_loss(self)
        return sigmoid / total

    def get_layers_primitive(self) -> List[LayerPrimitive]:
        return get_layers_primitive(self)

    def get_parameters_pruning_statistics(self) -> any:
        return get_layer_composite_pruning_statistics(self)

    def get_parameters_flipped_statistics(self) -> any:
        return get_layer_composite_flipped_statistics(self)

    def get_parameters_total_count(self) -> int:
        total = get_parameters_total_count(self)
        return total

    def load_pretrained_weights(self):
        # Load the pretrained ResNet-18 weights
        pretrained_state = torch.load(
            prefix_path_with_root(r"data\pretrained\resnet18_cifar10_95.bin"),
            map_location=get_device()
        )

        # Load weights for the initial convolutional layer
        self.conv1.weights.data.copy_(pretrained_state['conv1.weight'])
        print("Loaded weights for 'conv1'")

        # Load weights for the initial batch normalization layer
        self.bn1.weight.data.copy_(pretrained_state['bn1.weight'])
        self.bn1.bias.data.copy_(pretrained_state['bn1.bias'])
        self.bn1.running_mean.data.copy_(pretrained_state['bn1.running_mean'])
        self.bn1.running_var.data.copy_(pretrained_state['bn1.running_var'])
        print("Loaded weights for 'bn1'")

        # Function to map layer indices
        def map_layers(layer_num, block_num):
            pretrained_prefix = f'layer{layer_num}.{block_num - 1}'
            custom_prefix = f'layer{layer_num}_block{block_num}'
            return pretrained_prefix, custom_prefix

        # Loop over layers and blocks
        for layer_num in range(1, 5):  # Layers 1 to 4
            num_blocks = 2  # ResNet-18 has 2 blocks per layer
            for block_num in range(1, num_blocks + 1):
                pretrained_prefix, custom_prefix = map_layers(layer_num, block_num)

                # Load conv1 weights
                conv1_pretrained = f'{pretrained_prefix}.conv1.weight'
                conv1_custom = getattr(self, f'{custom_prefix}_conv1')
                conv1_custom.weights.data.copy_(pretrained_state[conv1_pretrained])

                # Load bn1 weights
                bn1_pretrained = f'{pretrained_prefix}.bn1'
                bn1_custom = getattr(self, f'{custom_prefix}_bn1')
                bn1_custom.weight.data.copy_(pretrained_state[f'{bn1_pretrained}.weight'])
                bn1_custom.bias.data.copy_(pretrained_state[f'{bn1_pretrained}.bias'])
                bn1_custom.running_mean.data.copy_(pretrained_state[f'{bn1_pretrained}.running_mean'])
                bn1_custom.running_var.data.copy_(pretrained_state[f'{bn1_pretrained}.running_var'])

                # Load conv2 weights
                conv2_pretrained = f'{pretrained_prefix}.conv2.weight'
                conv2_custom = getattr(self, f'{custom_prefix}_conv2')
                conv2_custom.weights.data.copy_(pretrained_state[conv2_pretrained])

                # Load bn2 weights
                bn2_pretrained = f'{pretrained_prefix}.bn2'
                bn2_custom = getattr(self, f'{custom_prefix}_bn2')
                bn2_custom.weight.data.copy_(pretrained_state[f'{bn2_pretrained}.weight'])
                bn2_custom.bias.data.copy_(pretrained_state[f'{bn2_pretrained}.bias'])
                bn2_custom.running_mean.data.copy_(pretrained_state[f'{bn2_pretrained}.running_mean'])
                bn2_custom.running_var.data.copy_(pretrained_state[f'{bn2_pretrained}.running_var'])

                print(f"Loaded weights for '{custom_prefix}'")

                # Load downsample layers if present
                downsample_key = f'{pretrained_prefix}.downsample.0.weight'
                if downsample_key in pretrained_state:
                    # Downsample convolutional layer
                    downsample_conv_pretrained = downsample_key
                    downsample_conv_custom = getattr(self, f'{custom_prefix}_downsample')
                    downsample_conv_custom.weights.data.copy_(pretrained_state[downsample_conv_pretrained])

                    # Downsample batch normalization layer
                    downsample_bn_pretrained = f'{pretrained_prefix}.downsample.1'
                    downsample_bn_custom = getattr(self, f'{custom_prefix}_downsample_bn')
                    downsample_bn_custom.weight.data.copy_(pretrained_state[f'{downsample_bn_pretrained}.weight'])
                    downsample_bn_custom.bias.data.copy_(pretrained_state[f'{downsample_bn_pretrained}.bias'])
                    downsample_bn_custom.running_mean.data.copy_(pretrained_state[f'{downsample_bn_pretrained}.running_mean'])
                    downsample_bn_custom.running_var.data.copy_(pretrained_state[f'{downsample_bn_pretrained}.running_var'])

                    print(f"Loaded downsample weights for '{custom_prefix}'")

        # Load weights for the fully connected layer, if sizes match
        fc_weight_key = 'fc.weight'
        fc_bias_key = 'fc.bias'
        if self.fc.weights.size() == pretrained_state[fc_weight_key].size():
            self.fc.weights.data.copy_(pretrained_state[fc_weight_key])
            self.fc.bias.data.copy_(pretrained_state[fc_bias_key])
            print("Loaded weights for 'fc'")
        else:
            print(f"Skipping 'fc' weights due to size mismatch.")

        print(f"Successfully loaded pretrained weights.")

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool1(x)

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
