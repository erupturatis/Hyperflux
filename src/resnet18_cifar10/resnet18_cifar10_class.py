from typing import List
import torch
import torch.nn as nn

from src.common_files_experiments.forward_functions import forward_pass_resnet18
from src.common_files_experiments.load_save import save_model_weights, load_model_weights
from src.resnet18_cifar10.resnet18_cifar10_attributes import \
    RESNET18_CIFAR10_REGISTERED_LAYERS_ATTRIBUTES, \
    RESNET18_CIFAR10_UNREGISTERED_LAYERS_ATTRIBUTES, RESNET18_CIFAR10_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING, \
    RESNET18_CIFAR10_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING
from src.infrastructure.constants import CONV2D_LAYER, FULLY_CONNECTED_LAYER, N_SCALER, PRUNED_MODELS_PATH
from src.infrastructure.layers import LayerConv2MaskImportance, ConfigsNetworkMasksImportance, LayerLinearMaskImportance, LayerComposite, LayerPrimitive, \
    get_layers_primitive, get_remaining_parameters_loss_masks_importance, get_layer_composite_pruning_statistics, ConfigsLayerConv2, \
    ConfigsLayerLinear, get_parameters_total_count


class Resnet18Cifar10(LayerComposite):
    def __init__(self, configs_network_masks: ConfigsNetworkMasksImportance):
        super(Resnet18Cifar10, self).__init__()
        self.registered_layers = []

        # hardcoded activations
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(
            output_size=(1,1)
        )

        print(RESNET18_CIFAR10_REGISTERED_LAYERS_ATTRIBUTES)
        for layer_attr in RESNET18_CIFAR10_REGISTERED_LAYERS_ATTRIBUTES:
            name = layer_attr['name']
            type_ = layer_attr['type']

            if type_ == CONV2D_LAYER:
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
            elif type_ == FULLY_CONNECTED_LAYER:
                print(layer_attr['name'], layer_attr['type'])
                print(layer_attr["in_features"], layer_attr["out_features"])
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
        total, remaining =  get_remaining_parameters_loss_masks_importance(self)
        return remaining / total

    def get_layers_primitive(self) -> List[LayerPrimitive]:
        return get_layers_primitive(self)

    def get_parameters_pruning_statistics(self) -> any:
        return get_layer_composite_pruning_statistics(self)

    def get_parameters_total_count(self) -> int:
        total = get_parameters_total_count(self)
        return total

    def forward(self, x):
        return forward_pass_resnet18(
            self=self,
            x=x,
            registered_layers=RESNET18_CIFAR10_REGISTERED_LAYERS_ATTRIBUTES,
            unregistered_layers=RESNET18_CIFAR10_UNREGISTERED_LAYERS_ATTRIBUTES
        )

    def save(self, name: str, folder:str):
        save_model_weights(
            model=self,
            model_name=name,
            folder_name=folder,
            custom_to_standard_mapping=RESNET18_CIFAR10_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING,
            skip_array=[]
        )

    def load(self, path: str, folder: str):
        load_model_weights(
            model=self,
            model_name=path,
            folder_name=folder,
            standard_to_custom_mapping=RESNET18_CIFAR10_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
            skip_array=[]
        )