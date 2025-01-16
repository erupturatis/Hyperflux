import torchvision.models as models
from typing import List
import torch
import torch.nn as nn

from src.common_files_experiments.load_save import save_model_weights, load_model_weights
from src.common_files_experiments.resnet18_small_images_functions import forward_pass_resnet18, ConfigsModelBaseResnet18
from src.common_files_experiments.resnet18_small_images_attributes import \
    RESNET18_SMALL_IMAGES_REGISTERED_LAYERS_ATTRIBUTES, \
    RESNET18_SMALL_IMAGES_UNREGISTERED_LAYERS_ATTRIBUTES, RESNET18_SMALL_IMAGES_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING, \
    RESNET18_SMALL_IMAGES_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING
from src.infrastructure.constants import CONV2D_LAYER, FULLY_CONNECTED_LAYER, N_SCALER, PRUNED_MODELS_PATH
from src.infrastructure.layers import LayerConv2MaskImportance, ConfigsNetworkMasksImportance, LayerLinearMaskImportance, LayerComposite, LayerPrimitive, \
    get_layers_primitive, get_remaining_parameters_loss_masks_importance, get_layer_composite_pruning_statistics, ConfigsLayerConv2, \
    ConfigsLayerLinear, get_layer_composite_flipped_statistics, get_parameters_total_count


class ModelBaseResnet18(LayerComposite):
    def __init__(self, configs_model_base_resnet: ConfigsModelBaseResnet18, configs_network_masks: ConfigsNetworkMasksImportance):
        super(ModelBaseResnet18, self).__init__()
        self.registered_layers = []

        # hardcoded activations
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(
            output_size=(1,1)
        )

        self.NUM_OUTPUT_CLASSES = configs_model_base_resnet.num_classes

        for layer_attr in RESNET18_SMALL_IMAGES_REGISTERED_LAYERS_ATTRIBUTES:
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

        for layer_attr in RESNET18_SMALL_IMAGES_UNREGISTERED_LAYERS_ATTRIBUTES:
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
        return remaining * N_SCALER

    def get_layers_primitive(self) -> List[LayerPrimitive]:
        return get_layers_primitive(self)

    def get_parameters_pruning_statistics(self) -> any:
        return get_layer_composite_pruning_statistics(self)

    def get_parameters_total_count(self) -> int:
        total = get_parameters_total_count(self)
        return total

    def forward(self, x):
        return forward_pass_resnet18(self, x)

    def save(self, name: str):
        save_model_weights(
            model=self,
            model_name=name,
            folder_name=PRUNED_MODELS_PATH,
            network_to_standard_mapping=RESNET18_SMALL_IMAGES_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING,
            skip_array=[]
        )

    def load(self, path: str):
        load_model_weights(
            model=self,
            model_name=path,
            folder_name=PRUNED_MODELS_PATH,
            standard_to_network_dict=RESNET18_SMALL_IMAGES_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
            skip_array=[]
        )