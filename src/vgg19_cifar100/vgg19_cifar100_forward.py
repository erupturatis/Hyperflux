from types import SimpleNamespace
import torch

from src.infrastructure.layers import LayerComposite
from src.vgg19_cifar100.vgg19_cifar100_attributes import (
    VGG19_CIFAR100_REGISTERED_LAYERS_ATTRIBUTES,
    VGG19_CIFAR100_UNREGISTERED_LAYERS_ATTRIBUTES
)

def forward_pass_vgg19_cifar100(self: 'LayerComposite', x: torch.Tensor) -> torch.Tensor:
    registered_layers_object = SimpleNamespace()
    for layer in VGG19_CIFAR100_REGISTERED_LAYERS_ATTRIBUTES:
        name = layer['name']
        layer_instance = getattr(self, name)
        setattr(registered_layers_object, name, layer_instance)

    unregistered_layers_object = SimpleNamespace()

    # Block 1
    x = registered_layers_object.conv1_1(x)
    x = torch.relu(x)
    x = registered_layers_object.conv1_2(x)
    x = torch.relu(x)
    x = self.maxpool(x)

    # Block 2
    x = registered_layers_object.conv2_1(x)
    x = torch.relu(x)
    x = registered_layers_object.conv2_2(x)
    x = torch.relu(x)
    x = self.maxpool(x)

    # Block 3
    x = registered_layers_object.conv3_1(x)
    x = torch.relu(x)
    x = registered_layers_object.conv3_2(x)
    x = torch.relu(x)
    x = registered_layers_object.conv3_3(x)
    x = torch.relu(x)
    x = registered_layers_object.conv3_4(x)
    x = torch.relu(x)
    x = self.maxpool(x)

    # Block 4
    x = registered_layers_object.conv4_1(x)
    x = torch.relu(x)
    x = registered_layers_object.conv4_2(x)
    x = torch.relu(x)
    x = registered_layers_object.conv4_3(x)
    x = torch.relu(x)
    x = registered_layers_object.conv4_4(x)
    x = torch.relu(x)
    x = self.maxpool(x)

    # Block 5
    x = registered_layers_object.conv5_1(x)
    x = torch.relu(x)
    x = registered_layers_object.conv5_2(x)
    x = torch.relu(x)
    x = registered_layers_object.conv5_3(x)
    x = torch.relu(x)
    x = registered_layers_object.conv5_4(x)
    x = torch.relu(x)
    x = self.maxpool(x)

    # Flatten the tensor
    x = torch.flatten(x, 1)  # Flatten all dimensions except batch

    # Fully Connected Layers
    x = registered_layers_object.fc1(x)
    x = torch.relu(x)
    x = registered_layers_object.fc2(x)
    x = torch.relu(x)
    x = registered_layers_object.fc3(x)

    return x
