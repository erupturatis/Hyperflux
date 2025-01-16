from typing import List
import torch
import torch.nn as nn
from dataclasses import dataclass

from src.imagenet1k_resnet50.common_resnet50 import load_model_weights
from src.imagenet1k_resnet50.resnet50_imagenet_attributes import RESNET50_IMAGENET_REGISTERED_LAYERS_ATTRIBUTES, \
    RESNET50_IMAGENET_UNREGISTERED_LAYERS_ATTRIBUTES
from src.infrastructure.constants import N_SCALER
from src.infrastructure.layers import LayerComposite, ConfigsNetworkMasksImportance, LayerConv2MaskImportance, \
    ConfigsLayerConv2, LayerLinearMaskImportance, ConfigsLayerLinear, LayerPrimitive, get_remaining_parameters_loss, \
    get_layers_primitive


@dataclass
class ConfigsModelBaseResnet50:
    num_classes: int

class ModelBaseResnet50(LayerComposite):
    def __init__(self, configs_model_base_resnet: ConfigsModelBaseResnet50, configs_network_masks: ConfigsNetworkMasksImportance):
        super(ModelBaseResnet50, self).__init__()
        self.registered_layers = []

        # Hardcoded activations
        self.relu = nn.ReLU(inplace=True)
        self.NUM_OUTPUT_CLASSES = configs_model_base_resnet.num_classes

        # Initialize registered layers
        for layer_attr in RESNET50_IMAGENET_REGISTERED_LAYERS_ATTRIBUTES:
            name = layer_attr['name']
            type_ = layer_attr['type']

            if type_ == 'LayerConv2':
                layer = LayerConv2MaskImportance(
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
                layer = LayerLinearMaskImportance(
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

        # Initialize unregistered layers
        for layer_attr in RESNET50_IMAGENET_UNREGISTERED_LAYERS_ATTRIBUTES:
            name = layer_attr['name']
            type_ = layer_attr['type']

            if type_ == 'BatchNorm2d':
                layer = nn.BatchNorm2d(
                    num_features=layer_attr['num_features']
                )
            elif type_ == 'MaxPool2d':
                layer = nn.MaxPool2d(
                    kernel_size=layer_attr['kernel_size'],
                    stride=layer_attr['stride'],
                    padding=layer_attr['padding']
                )
            elif type_ == 'AdaptiveAvgPool2d':
                layer = nn.AdaptiveAvgPool2d(
                    output_size=layer_attr['output_size']
                )
            else:
                raise ValueError(f"Unsupported unregistered layer type: {type_}")

            setattr(self, name, layer)

        # Initialize additional layers if any
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def get_remaining_parameters_loss(self) -> torch.Tensor:
        total, sigmoid = get_remaining_parameters_loss(self)
        return sigmoid * N_SCALER

    def get_layers_primitive(self) -> List[LayerPrimitive]:
        return get_layers_primitive(self)

    def load_weights(self, model, skip_array = []):
        load_model_weights(model, skip_array)

    def save_weights(self, path : str):
        save_model_weights(self, path, skip_array = [])
        
    def forward(self, x):
        return forward_pass_resnet50(self, x)
    