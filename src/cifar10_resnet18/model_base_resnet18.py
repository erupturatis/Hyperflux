from typing import TypedDict
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.cifar10_resnet18.common_resnet18 import forward_pass_resnet18, load_model_weights, save_model_weights, \
    ConfigsModelBaseResnet18
from src.cifar10_resnet18.model_resnet18_attributes import RESNET18_CIFAR10_REGISTERED_LAYERS_ATTRIBUTES, \
    RESNET18_CIFAR10_UNREGISTERED_LAYERS_ATTRIBUTES
from src.others import get_device, prefix_path_with_root
from src.blocks_resnet import BlockResnet, ConfigsBlockResnet
from src.layers import LayerConv2, ConfigsNetworkMasks, LayerLinear, LayerComposite, LayerPrimitive, \
    get_layers_primitive, get_remaining_parameters_loss, get_layer_composite_pruning_statistics, ConfigsLayerConv2, \
    ConfigsLayerLinear, get_layer_composite_flipped_statistics, get_parameters_total_count
from dataclasses import dataclass


class ModelBaseResnet18(LayerComposite):
    def __init__(self, configs_model_base_resnet: ConfigsModelBaseResnet18, configs_network_masks: ConfigsNetworkMasks):
        super(ModelBaseResnet18, self).__init__()
        self.registered_layers = []

        # hardcoded activations
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(
            output_size=(1,1)
        )

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
            else:
                raise ValueError(f"Unsupported unregistered layer type: {type_}")

            setattr(self, name, layer)


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

    def forward(self, x):
        return forward_pass_resnet18(self, x)

    def save(self, path: str):
        save_model_weights(self, path, skip_array=[])

    def load(self, path: str):
        load_model_weights(self, path, skip_array=[])

    def load_pretrained_weights_DEPRECATED(self):
        """
        THIS METHOD IS DEPRECATED. USE CAREFULLY
        """
        pretrained_state = torch.load(
            prefix_path_with_root(r"data\pretrained\resnet18_cifar10_95.bin"),
            map_location=get_device()
        )

        # load weights for the initial convolutional layer
        self.conv1.weights.data.copy_(pretrained_state['conv1.weight'])
        print("loaded weights for 'conv1'")

        # load weights for the initial batch normalization layer
        self.bn1.weights.data.copy_(pretrained_state['bn1.weight'])
        self.bn1.bias_enabled.data.copy_(pretrained_state['bn1.bias'])
        self.bn1.running_mean.data.copy_(pretrained_state['bn1.running_mean'])
        self.bn1.running_var.data.copy_(pretrained_state['bn1.running_var'])
        print("loaded weights for 'bn1'")

        # function to map layer indices
        def map_layers(layer_num, block_num):
            pretrained_prefix = f'layer{layer_num}.{block_num - 1}'
            custom_prefix = f'layer{layer_num}_block{block_num}'
            return pretrained_prefix, custom_prefix

        # loop over layers and blocks
        for layer_num in range(1, 5):  # layers 1 to 4
            num_blocks = 2  # resnet-18 has 2 blocks per layer
            for block_num in range(1, num_blocks + 1):
                pretrained_prefix, custom_prefix = map_layers(layer_num, block_num)

                # load conv1 weights
                conv1_pretrained = f'{pretrained_prefix}.conv1.weight'
                conv1_custom = getattr(self, f'{custom_prefix}_conv1')
                conv1_custom.weights.data.copy_(pretrained_state[conv1_pretrained])

                # load bn1 weights
                bn1_pretrained = f'{pretrained_prefix}.bn1'
                bn1_custom = getattr(self, f'{custom_prefix}_bn1')
                bn1_custom.weights.data.copy_(pretrained_state[f'{bn1_pretrained}.weight'])
                bn1_custom.bias_enabled.data.copy_(pretrained_state[f'{bn1_pretrained}.bias'])
                bn1_custom.running_mean.data.copy_(pretrained_state[f'{bn1_pretrained}.running_mean'])
                bn1_custom.running_var.data.copy_(pretrained_state[f'{bn1_pretrained}.running_var'])

                # load conv2 weights
                conv2_pretrained = f'{pretrained_prefix}.conv2.weight'
                conv2_custom = getattr(self, f'{custom_prefix}_conv2')
                conv2_custom.weights.data.copy_(pretrained_state[conv2_pretrained])

                # load bn2 weights
                bn2_pretrained = f'{pretrained_prefix}.bn2'
                bn2_custom = getattr(self, f'{custom_prefix}_bn2')
                bn2_custom.weights.data.copy_(pretrained_state[f'{bn2_pretrained}.weight'])
                bn2_custom.bias_enabled.data.copy_(pretrained_state[f'{bn2_pretrained}.bias'])
                bn2_custom.running_mean.data.copy_(pretrained_state[f'{bn2_pretrained}.running_mean'])
                bn2_custom.running_var.data.copy_(pretrained_state[f'{bn2_pretrained}.running_var'])

                print(f"loaded weights for '{custom_prefix}'")

                # load downsample layers if present
                downsample_key = f'{pretrained_prefix}.downsample.0.weight'
                if downsample_key in pretrained_state:
                    # downsample convolutional layer
                    downsample_conv_pretrained = downsample_key
                    downsample_conv_custom = getattr(self, f'{custom_prefix}_downsample')
                    downsample_conv_custom.weights.data.copy_(pretrained_state[downsample_conv_pretrained])

                    # downsample batch normalization layer
                    downsample_bn_pretrained = f'{pretrained_prefix}.downsample.1'
                    downsample_bn_custom = getattr(self, f'{custom_prefix}_downsample_bn')
                    downsample_bn_custom.weights.data.copy_(pretrained_state[f'{downsample_bn_pretrained}.weight'])
                    downsample_bn_custom.bias_enabled.data.copy_(pretrained_state[f'{downsample_bn_pretrained}.bias'])
                    downsample_bn_custom.running_mean.data.copy_(pretrained_state[f'{downsample_bn_pretrained}.running_mean'])
                    downsample_bn_custom.running_var.data.copy_(pretrained_state[f'{downsample_bn_pretrained}.running_var'])

                    print(f"loaded downsample weights for '{custom_prefix}'")

        # load weights for the fully connected layer, if sizes match
        fc_weight_key = 'fc.weight'
        fc_bias_key = 'fc.bias'
        if self.fc.weights.size() == pretrained_state[fc_weight_key].size():
            self.fc.weights.data.copy_(pretrained_state[fc_weight_key])
            self.fc.bias_enabled.data.copy_(pretrained_state[fc_bias_key])
            print("loaded weights for 'fc'")
        else:
            print(f"skipping 'fc' weights due to size mismatch.")

        print(f"successfully loaded pretrained weights.")
