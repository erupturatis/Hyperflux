from typing import List
import torch
import torch.nn as nn
from src.common_files_experiments.load_save import save_model_weights, load_model_weights
from src.imagenet1k_resnet50.resnet50_imagenet_forward import forward_pass_resnet50_imagenet
from src.common_files_experiments.resnet50_vanilla_attributes import RESNET50_VANILLA_REGISTERED_LAYERS_ATTRIBUTES, \
    RESNET50_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES, RESNET50_VANILLA_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING, \
    RESNET50_VANILLA_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING
from src.infrastructure.constants import N_SCALER, PRUNED_MODELS_PATH
from src.infrastructure.layers import LayerComposite, ConfigsNetworkMasksImportance, LayerConv2MaskImportance, \
    ConfigsLayerConv2, LayerLinearMaskImportance, ConfigsLayerLinear, LayerPrimitive, get_remaining_parameters_loss, \
    get_layers_primitive


class Resnet50Imagenet(LayerComposite):
    def __init__(self, configs_network_masks: ConfigsNetworkMasksImportance):
        super(Resnet50Imagenet, self).__init__()
        self.registered_layers = []

        # Hardcoded activations
        self.relu = nn.ReLU(inplace=True)

        # Initialize registered layers
        for layer_attr in RESNET50_VANILLA_REGISTERED_LAYERS_ATTRIBUTES:
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
        for layer_attr in RESNET50_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES:
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

    def forward(self, x):
        return forward_pass_resnet50_imagenet(self, x)

    def save(self, name: str):
        save_model_weights(
            model=self,
            model_name=name,
            folder_name=PRUNED_MODELS_PATH,
            custom_to_standard_mapping=RESNET50_VANILLA_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING,
            skip_array=[]
        )

    def load(self, path: str):
        load_model_weights(
            model=self,
            model_name=path,
            folder_name=PRUNED_MODELS_PATH,
            standard_to_custom_mapping=RESNET50_VANILLA_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
            skip_array=[]
        )

