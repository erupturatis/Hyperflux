from types import SimpleNamespace
import torch
from src.resnet50_cifar10.resnet50_cifar10_attributes import RESNET50_CIFAR10_REGISTERED_LAYERS_ATTRIBUTES, \
    RESNET50_CIFAR10_UNREGISTERED_LAYERS_ATTRIBUTES

def forward_pass_resnet50_cifar10(self: 'LayerComposite', x: torch.Tensor) -> torch.Tensor:
    registered_layers_object = SimpleNamespace()
    for layer_attr in RESNET50_CIFAR10_REGISTERED_LAYERS_ATTRIBUTES:
        name = layer_attr['name']
        layer = getattr(self, name)
        setattr(registered_layers_object, name, layer)

    unregistered_layers_object = SimpleNamespace()
    for layer_attr in RESNET50_CIFAR10_UNREGISTERED_LAYERS_ATTRIBUTES:
        name = layer_attr['name']
        layer = getattr(self, name)
        setattr(unregistered_layers_object, name, layer)

    x = registered_layers_object.conv1(x)
    x = unregistered_layers_object.bn1(x)
    x = self.relu(x)
    x = unregistered_layers_object.maxpool1(x)

    layers_config = [
        {'layer_num': 1, 'num_blocks': 3},
        {'layer_num': 2, 'num_blocks': 4},
        {'layer_num': 3, 'num_blocks': 6},
        {'layer_num': 4, 'num_blocks': 3},
    ]

    # Process each layer
    for layer_info in layers_config:
        layer_num = layer_info['layer_num']
        num_blocks = layer_info['num_blocks']
        x = _forward_layer(self, x, layer_num, num_blocks, registered_layers_object, unregistered_layers_object)

    x = unregistered_layers_object.avgpool(x)
    x = torch.flatten(x, 1)
    x = registered_layers_object.fc(x)

    return x

def _forward_layer(self, x, layer_num, num_blocks, registered_layers_object, unregistered_layers_object):
    for block_num in range(1, num_blocks + 1):
        identity = x

        downsample_name = f'layer{layer_num}_block{block_num}_downsample'
        if hasattr(registered_layers_object, downsample_name):
            downsample_conv = getattr(registered_layers_object, downsample_name)
            downsample_bn = getattr(unregistered_layers_object, f'{downsample_name}_bn')
            identity = downsample_bn(downsample_conv(x))

        out = getattr(registered_layers_object, f'layer{layer_num}_block{block_num}_conv1')(x)
        out = getattr(unregistered_layers_object, f'layer{layer_num}_block{block_num}_bn1')(out)
        out = self.relu(out)

        out = getattr(registered_layers_object, f'layer{layer_num}_block{block_num}_conv2')(out)
        out = getattr(unregistered_layers_object, f'layer{layer_num}_block{block_num}_bn2')(out)
        out = self.relu(out)

        out = getattr(registered_layers_object, f'layer{layer_num}_block{block_num}_conv3')(out)
        out = getattr(unregistered_layers_object, f'layer{layer_num}_block{block_num}_bn3')(out)

        out += identity
        out = self.relu(out)

        x = out

    return x

