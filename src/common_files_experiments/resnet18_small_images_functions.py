from types import SimpleNamespace
from typing import TYPE_CHECKING
import torch
import torch.nn as nn

from src.infrastructure.constants import BASELINE_MODELS_PATH
from src.infrastructure.layers import LayerComposite, LayerPrimitive
from typing import List
from src.infrastructure.others import prefix_path_with_root
from dataclasses import dataclass
if TYPE_CHECKING:
    from src.common_files_experiments.resnet18_small_images_class import ModelBaseResnet18

from src.common_files_experiments.resnet18_small_images_attributes import RESNET18_SMALL_IMAGES_REGISTERED_LAYERS_ATTRIBUTES, \
    RESNET18_SMALL_IMAGES_UNREGISTERED_LAYERS_ATTRIBUTES, RESNET18_SMALL_IMAGES_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING, \
    RESNET18_SMALL_IMAGES_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING


@dataclass
class ConfigsModelBaseResnet18:
    num_classes: int

def forward_pass_resnet18(self: 'LayerComposite', x: torch.Tensor) -> torch.Tensor:
    # Ensures all layers used in forward are registered in these 2 arrays
    registered_layers_object = SimpleNamespace()
    for layer in RESNET18_SMALL_IMAGES_REGISTERED_LAYERS_ATTRIBUTES:
        name = layer['name']
        layer = getattr(self, name)
        setattr(registered_layers_object, name, layer)

    unregistered_layers_object = SimpleNamespace()
    for layer in RESNET18_SMALL_IMAGES_UNREGISTERED_LAYERS_ATTRIBUTES:
        name = layer['name']
        layer = getattr(self, name)
        setattr(unregistered_layers_object, name, layer)

    # Initial layers
    x = registered_layers_object.conv1(x)
    x = unregistered_layers_object.bn1(x)
    x = self.relu(x)

    # Layer 1
    # Block 1
    identity = x
    out = registered_layers_object.layer1_block1_conv1(x)
    out = unregistered_layers_object.layer1_block1_bn1(out)
    out = self.relu(out)
    out = registered_layers_object.layer1_block1_conv2(out)
    out = unregistered_layers_object.layer1_block1_bn2(out)
    out += identity
    out = self.relu(out)

    # Block 2
    identity = out
    out = registered_layers_object.layer1_block2_conv1(out)
    out = unregistered_layers_object.layer1_block2_bn1(out)
    out = self.relu(out)
    out = registered_layers_object.layer1_block2_conv2(out)
    out = unregistered_layers_object.layer1_block2_bn2(out)
    out += identity
    out = self.relu(out)

    # Layer 2
    # Block 1 with downsampling
    identity = out
    out = registered_layers_object.layer2_block1_conv1(out)
    out = unregistered_layers_object.layer2_block1_bn1(out)
    out = self.relu(out)
    out = registered_layers_object.layer2_block1_conv2(out)
    out = unregistered_layers_object.layer2_block1_bn2(out)

    identity = registered_layers_object.layer2_block1_downsample(identity)
    identity = unregistered_layers_object.layer2_block1_downsample_bn(identity)

    out += identity
    out = self.relu(out)

    # Block 2
    identity = out
    out = registered_layers_object.layer2_block2_conv1(out)
    out = unregistered_layers_object.layer2_block2_bn1(out)
    out = self.relu(out)
    out = registered_layers_object.layer2_block2_conv2(out)
    out = unregistered_layers_object.layer2_block2_bn2(out)
    out += identity
    out = self.relu(out)

    # Layer 3
    # Block 1 with downsampling
    identity = out
    out = registered_layers_object.layer3_block1_conv1(out)
    out = unregistered_layers_object.layer3_block1_bn1(out)
    out = self.relu(out)
    out = registered_layers_object.layer3_block1_conv2(out)
    out = unregistered_layers_object.layer3_block1_bn2(out)

    identity = registered_layers_object.layer3_block1_downsample(identity)
    identity = unregistered_layers_object.layer3_block1_downsample_bn(identity)

    out += identity
    out = self.relu(out)

    # Block 2
    identity = out
    out = registered_layers_object.layer3_block2_conv1(out)
    out = unregistered_layers_object.layer3_block2_bn1(out)
    out = self.relu(out)
    out = registered_layers_object.layer3_block2_conv2(out)
    out = unregistered_layers_object.layer3_block2_bn2(out)
    out += identity
    out = self.relu(out)

    # Layer 4
    # Block 1 with downsampling
    identity = out
    out = registered_layers_object.layer4_block1_conv1(out)
    out = unregistered_layers_object.layer4_block1_bn1(out)
    out = self.relu(out)
    out = registered_layers_object.layer4_block1_conv2(out)
    out = unregistered_layers_object.layer4_block1_bn2(out)

    identity = registered_layers_object.layer4_block1_downsample(identity)
    identity = unregistered_layers_object.layer4_block1_downsample_bn(identity)

    out += identity
    out = self.relu(out)

    # Block 2
    identity = out
    out = registered_layers_object.layer4_block2_conv1(out)
    out = unregistered_layers_object.layer4_block2_bn1(out)
    out = self.relu(out)
    out = registered_layers_object.layer4_block2_conv2(out)
    out = unregistered_layers_object.layer4_block2_bn2(out)
    out += identity
    out = self.relu(out)

    # Final layers
    out = self.avgpool(out)
    out = torch.flatten(out, 1)
    out = registered_layers_object.fc(out)

    return out

def save_model_weights_resnet18_cifar10(model: 'LayerComposite', filepath: str, skip_array: List = []):
    """
    Saves the model weights in a format compatible with standard ResNet18 implementations.

    Args:
        model (ModelBaseResnet18): The custom ResNet18 model instance.
        filepath (str): The path where the weights will be saved.
    """
    import torch
    filepath = prefix_path_with_root(filepath)

    # Initialize an empty state_dict
    state_dict = {}

    # Iterate over the mapping array
    for mapping in RESNET18_SMALL_IMAGES_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING:
        custom_name = mapping['custom_name']
        standard_name = mapping['standard_name']
        if custom_name in skip_array:
            continue

        # Retrieve the layer using only the attributes array
        layer = getattr(model, custom_name, None)
        if layer is None:
            print(f"Layer '{custom_name}' not found in the model.")
            continue

        # Handle BatchNorm layers
        if isinstance(layer, nn.BatchNorm2d):
            state_dict[f'{standard_name}.weight'] = layer.weight.data.clone()
            state_dict[f'{standard_name}.bias'] = layer.bias.data.clone()
            state_dict[f'{standard_name}.running_mean'] = layer.running_mean.data.clone()
            state_dict[f'{standard_name}.running_var'] = layer.running_var.data.clone()
            state_dict[f'{standard_name}.num_batches_tracked'] = layer.num_batches_tracked.clone()

        # Handle Conv2d layers
        elif isinstance(layer, LayerPrimitive):
            state_dict[standard_name] = layer.get_applied_weights().data.clone()
            if layer.get_bias_enabled():
                bias_name = standard_name.replace('.weight', '.bias')
                state_dict[bias_name] = layer.bias.data.clone()

        else:
            print(f"Unhandled layer type for layer '{custom_name}': {type(layer)}")

    # Save the state_dict
    torch.save(state_dict, filepath)
    print(f"Model weights saved to {filepath}.")

def load_model_weights_resnet18_cifar10(model: 'LayerComposite', model_dict, skip_array: List = []):
    state_dict = model_dict

    # Iterate over the mapping array
    for mapping in RESNET18_SMALL_IMAGES_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING:
        standard_name = mapping['standard_name']
        custom_name = mapping['custom_name']
        if custom_name in skip_array:
            continue

        # Retrieve the layer using only the attributes array
        layer = getattr(model, custom_name, None)
        if layer is None:
            print(f"Layer '{custom_name}' not found in the model.")
            continue

        # Handle BatchNorm layers
        if isinstance(layer, nn.BatchNorm2d):
            layer.weight.data.copy_(state_dict[f'{standard_name}.weight'])
            layer.bias.data.copy_(state_dict[f'{standard_name}.bias'])
            layer.running_mean.data.copy_(state_dict[f'{standard_name}.running_mean'])
            layer.running_var.data.copy_(state_dict[f'{standard_name}.running_var'])
            layer.num_batches_tracked.copy_(state_dict[f'{standard_name}.num_batches_tracked'])

        # Handle Conv2d layers
        elif isinstance(layer, LayerPrimitive):
            layer.weights.data.copy_(state_dict[standard_name])
            if layer.get_bias_enabled():
                bias_name = standard_name.replace('.weight', '.bias')
                layer.bias.data.copy_(state_dict[bias_name])

        else:
            print(f"Unhandled layer type for layer '{custom_name}': {type(layer)}")

def load_model_weights_resnet18_cifar10_from_path(model: 'LayerComposite', model_name: str, skip_array: List = []):
    filepath = BASELINE_MODELS_PATH + "/" + model_name
    filepath = prefix_path_with_root(filepath)
    state_dict = torch.load(filepath)
    load_model_weights_resnet18_cifar10(model, state_dict, skip_array)

