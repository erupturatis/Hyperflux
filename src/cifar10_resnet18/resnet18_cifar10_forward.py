from types import SimpleNamespace
from typing import TYPE_CHECKING
import torch

from src.infrastructure.layers import LayerComposite

if TYPE_CHECKING:
    pass

from src.cifar10_resnet18.resnet18_cifar10_attributes import RESNET18_CIFAR10_REGISTERED_LAYERS_ATTRIBUTES, \
    RESNET18_CIFAR10_UNREGISTERED_LAYERS_ATTRIBUTES


def forward_pass_resnet18_cifar10(self: 'LayerComposite', x: torch.Tensor) -> torch.Tensor:
    # Ensures all layers used in forward are registered in these 2 arrays
    registered_layers_object = SimpleNamespace()
    for layer in RESNET18_CIFAR10_REGISTERED_LAYERS_ATTRIBUTES:
        name = layer['name']
        layer = getattr(self, name)
        setattr(registered_layers_object, name, layer)

    unregistered_layers_object = SimpleNamespace()
    for layer in RESNET18_CIFAR10_UNREGISTERED_LAYERS_ATTRIBUTES:
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

