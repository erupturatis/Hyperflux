from typing import TypedDict
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.cifar10_resnet18.common_resnet18 import forward_pass_resnet18, load_model_weights_resnet18_cifar10, save_model_weights_resnet18_cifar10, \
    ConfigsModelBaseResnet18
from src.cifar10_resnet18.model_resnet18_attributes import RESNET18_CIFAR10_REGISTERED_LAYERS_ATTRIBUTES, \
    RESNET18_CIFAR10_UNREGISTERED_LAYERS_ATTRIBUTES
from src.others import get_device, prefix_path_with_root
from src.blocks_resnet import BlockResnet, ConfigsBlockResnet
from src.layers import LayerConv2, ConfigsNetworkMasks, LayerLinear, LayerComposite, LayerPrimitive, \
    get_layers_primitive, get_remaining_parameters_loss, get_layer_composite_pruning_statistics, ConfigsLayerConv2, \
    ConfigsLayerLinear, get_layer_composite_flipped_statistics, get_parameters_total_count, LayerConv2Vanilla, \
    LayerLinearVanilla, get_layer_composite_pruning_statistics_vanilla
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelVanillaResnet18(LayerComposite):
    def __init__(self, configs_model_base_resnet: ConfigsModelBaseResnet18):
        super(ModelVanillaResnet18, self).__init__()
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
                layer = LayerConv2Vanilla(
                    ConfigsLayerConv2(
                        in_channels=layer_attr['in_channels'],
                        out_channels=layer_attr['out_channels'],
                        kernel_size=layer_attr['kernel_size'],
                        stride=layer_attr['stride'],
                        padding=layer_attr['padding'],
                        bias_enabled=layer_attr['bias_enabled']
                    )
                )
            elif type_ == 'LayerLinear':
                layer = LayerLinearVanilla(
                    ConfigsLayerLinear(
                        in_features=layer_attr['in_features'],
                        out_features=layer_attr['out_features']
                    )
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

        self.load("saved_models/model_5epochs.pth")

    def get_layers_primitive(self) -> List[LayerPrimitive]:
        return get_layers_primitive(self)

    def get_parameters_pruning_statistics(self) -> any:
        return get_layer_composite_pruning_statistics_vanilla(self)

    def get_parameters_total_count(self) -> int:
        total = get_parameters_total_count(self)
        return total

    def forward(self, x):
        return forward_pass_resnet18(self, x)

    def save(self, path: str):
        save_model_weights_resnet18_cifar10(self, path, skip_array=[])

    def load(self, path: str):
        load_model_weights_resnet18_cifar10(self, path, skip_array=[])
