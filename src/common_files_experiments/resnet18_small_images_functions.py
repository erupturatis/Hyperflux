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

