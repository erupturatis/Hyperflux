import torch
from typing import List

from src.cifar10_resnet18.model_functions import forward_pass_resnet18
from src.infrastructure.layers import ConfigsNetworkMasksImportance, LayerLinearMaskImportance, MaskPruningFunctionSigmoid, ConfigsLayerLinear, \
    get_remaining_parameters_loss_masks_importance, get_layer_composite_flipped_statistics, get_layer_composite_pruning_statistics, \
    LayerPrimitive, LayerComposite, get_layers_primitive
from src.infrastructure.others import get_device
from src.infrastructure.constants import WEIGHTS_PRUNING_ATTR, WEIGHTS_FLIPPING_ATTR, FULLY_CONNECTED_LAYER
from src.mnist_lenet300.model_attributes import LENET300_MNIST_UNREGISTERED_LAYERS_ATTRIBUTES, \
    LENET300_MNIST_REGISTERED_LAYERS_ATTRIBUTES

class ModelLenet300(LayerComposite):
    def __init__(self, config_network_mask: ConfigsNetworkMasksImportance):
        super(ModelLenet300, self).__init__()
        self.registered_layers = []

        for layer_attr in LENET300_MNIST_REGISTERED_LAYERS_ATTRIBUTES:
            name = layer_attr['name']
            type_ = layer_attr['type']

            if type_ == FULLY_CONNECTED_LAYER:
                layer = LayerLinearMaskImportance(
                    configs_linear=ConfigsLayerLinear(
                        in_features=layer_attr['in_features'],
                        out_features=layer_attr['out_features']
                    ),
                    configs_network=config_network_mask,
                    bias=layer_attr.get('bias_enabled', True)
                )
            else:
                raise ValueError(f"Unsupported registered layer type: {type_}")

            setattr(self, name, layer)
            self.registered_layers.append(layer)

    def get_remaining_parameters_loss(self) -> torch.Tensor:
        total, sigmoid = get_remaining_parameters_loss_masks_importance(self)
        return sigmoid / total

    def get_layers_primitive(self) -> List[LayerPrimitive]:
        return get_layers_primitive(self)

    def get_parameters_pruning_statistics(self) -> any:
        return get_layer_composite_pruning_statistics(self)

    def get_parameters_flipped_statistics(self) -> any:
        return get_layer_composite_flipped_statistics(self)

    def forward(self, x, inference=False):
        return forward_pass_resnet18(self, x, inference)
