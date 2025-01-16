from src.common_files_experiments.attributes_mutations import Mutation, mutate_attributes
from src.common_files_experiments.resnet18_vanilla_attributes import RESNET18_VANILLA_REGISTERED_LAYERS_ATTRIBUTES, \
    RESNET18_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES, RESNET18_VANILLA_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING, \
    RESNET18_VANILLA_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING
from src.infrastructure.constants import CONV2D_LAYER, FULLY_CONNECTED_LAYER

replace_conv1 = Mutation(
    field_identified='name',
    value_in_field='conv1',
    action='replace',
    replacement_dict={
        "name": "conv1",
        "type": CONV2D_LAYER,
        "in_channels": 3,
        "out_channels": 64,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "bias_enabled": False
    }
)

replace_fc = Mutation(
    field_identified='name',
    value_in_field='fc',
    action='replace',
    replacement_dict={
        "name": "fc",
        "type": FULLY_CONNECTED_LAYER,
        "in_features": 512,
        "out_features": 10
    }
)

cifar10_registered_mutations = [
    replace_conv1,
    replace_fc
]

RESNET18_CIFAR10_REGISTERED_LAYERS_ATTRIBUTES = mutate_attributes(
    attributes=RESNET18_VANILLA_REGISTERED_LAYERS_ATTRIBUTES,
    mutations=cifar10_registered_mutations
)

RESNET18_CIFAR10_UNREGISTERED_LAYERS_ATTRIBUTES = mutate_attributes(
    attributes=RESNET18_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES,
    mutations=[]  # No mutations to apply
)

RESNET18_CIFAR10_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING = mutate_attributes(
    attributes= RESNET18_VANILLA_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
    mutations=[]  # No mutations to apply
)

RESNET18_CIFAR10_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING = mutate_attributes(
    attributes= RESNET18_VANILLA_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING,
    mutations=[]  # No mutations to apply
)


# RESNET18_CIFAR10_REGISTERED_LAYERS_ATTRIBUTES = [
#     # Initial convolutional layer
#     {"name": "conv1", "type": CONV2D_LAYER, "in_channels": 3, "out_channels": 64,
#      "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
#
#     # Layer 1 - Block 1
#     {"name": "layer1_block1_conv1", "type": CONV2D_LAYER, "in_channels": 64,
#      "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
#     {"name": "layer1_block1_conv2", "type": CONV2D_LAYER, "in_channels": 64,
#      "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
#
#     # Layer 1 - Block 2
#     {"name": "layer1_block2_conv1", "type": CONV2D_LAYER, "in_channels": 64,
#      "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
#     {"name": "layer1_block2_conv2", "type": CONV2D_LAYER, "in_channels": 64,
#      "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
#
#     # Layer 2 - Block 1
#     {"name": "layer2_block1_conv1", "type": CONV2D_LAYER, "in_channels": 64,
#      "out_channels": 128, "kernel_size": 3, "stride": 2, "padding": 1, "bias_enabled": False},
#     {"name": "layer2_block1_conv2", "type": CONV2D_LAYER, "in_channels": 128,
#      "out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
#     {"name": "layer2_block1_downsample", "type": CONV2D_LAYER, "in_channels": 64,
#      "out_channels": 128, "kernel_size": 1, "stride": 2, "padding": 0, "bias_enabled": False},
#
#     # Layer 2 - Block 2
#     {"name": "layer2_block2_conv1", "type": CONV2D_LAYER, "in_channels": 128,
#      "out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
#     {"name": "layer2_block2_conv2", "type": CONV2D_LAYER, "in_channels": 128,
#      "out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
#
#     # Layer 3 - Block 1
#     {"name": "layer3_block1_conv1", "type": CONV2D_LAYER, "in_channels": 128,
#      "out_channels": 256, "kernel_size": 3, "stride": 2, "padding": 1, "bias_enabled": False},
#     {"name": "layer3_block1_conv2", "type": CONV2D_LAYER, "in_channels": 256,
#      "out_channels": 256, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
#     {"name": "layer3_block1_downsample", "type": CONV2D_LAYER, "in_channels": 128,
#      "out_channels": 256, "kernel_size": 1, "stride": 2, "padding": 0, "bias_enabled": False},
#
#     # Layer 3 - Block 2
#     {"name": "layer3_block2_conv1", "type": CONV2D_LAYER, "in_channels": 256,
#      "out_channels": 256, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
#     {"name": "layer3_block2_conv2", "type": CONV2D_LAYER, "in_channels": 256,
#      "out_channels": 256, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
#
#     # Layer 4 - Block 1
#     {"name": "layer4_block1_conv1", "type": CONV2D_LAYER, "in_channels": 256,
#      "out_channels": 512, "kernel_size": 3, "stride": 2, "padding": 1, "bias_enabled": False},
#     {"name": "layer4_block1_conv2", "type": CONV2D_LAYER, "in_channels": 512,
#      "out_channels": 512, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
#     {"name": "layer4_block1_downsample", "type": CONV2D_LAYER, "in_channels": 256,
#      "out_channels": 512, "kernel_size": 1, "stride": 2, "padding": 0, "bias_enabled": False},
#
#     # Layer 4 - Block 2
#     {"name": "layer4_block2_conv1", "type": CONV2D_LAYER, "in_channels": 512,
#      "out_channels": 512, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
#     {"name": "layer4_block2_conv2", "type": CONV2D_LAYER, "in_channels": 512,
#      "out_channels": 512, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
#
#     # Fully connected layer
#     {"name": "fc", "type": FULLY_CONNECTED_LAYER, "in_features": 512, "out_features": 10}
# ]
#
#
# # Unregistered layers (e.g., batch norms, activations, pooling)
# RESNET18_CIFAR10_UNREGISTERED_LAYERS_ATTRIBUTES = [
#     # Initial batch normalization
#     {"name": "bn1", "type": "BatchNorm2d", "num_features": 64},
#
#     # Layer 1 - Block 1
#     {"name": "layer1_block1_bn1", "type": "BatchNorm2d", "num_features": 64},
#     {"name": "layer1_block1_bn2", "type": "BatchNorm2d", "num_features": 64},
#
#     # Layer 1 - Block 2
#     {"name": "layer1_block2_bn1", "type": "BatchNorm2d", "num_features": 64},
#     {"name": "layer1_block2_bn2", "type": "BatchNorm2d", "num_features": 64},
#
#     # Layer 2 - Block 1
#     {"name": "layer2_block1_bn1", "type": "BatchNorm2d", "num_features": 128},
#     {"name": "layer2_block1_bn2", "type": "BatchNorm2d", "num_features": 128},
#     {"name": "layer2_block1_downsample_bn", "type": "BatchNorm2d", "num_features": 128},
#
#     # Layer 2 - Block 2
#     {"name": "layer2_block2_bn1", "type": "BatchNorm2d", "num_features": 128},
#     {"name": "layer2_block2_bn2", "type": "BatchNorm2d", "num_features": 128},
#
#     # Layer 3 - Block 1
#     {"name": "layer3_block1_bn1", "type": "BatchNorm2d", "num_features": 256},
#     {"name": "layer3_block1_bn2", "type": "BatchNorm2d", "num_features": 256},
#     {"name": "layer3_block1_downsample_bn", "type": "BatchNorm2d", "num_features": 256},
#
#     # Layer 3 - Block 2
#     {"name": "layer3_block2_bn1", "type": "BatchNorm2d", "num_features": 256},
#     {"name": "layer3_block2_bn2", "type": "BatchNorm2d", "num_features": 256},
#
#     # Layer 4 - Block 1
#     {"name": "layer4_block1_bn1", "type": "BatchNorm2d", "num_features": 512},
#     {"name": "layer4_block1_bn2", "type": "BatchNorm2d", "num_features": 512},
#     {"name": "layer4_block1_downsample_bn", "type": "BatchNorm2d", "num_features": 512},
#
#     # Layer 4 - Block 2
#     {"name": "layer4_block2_bn1", "type": "BatchNorm2d", "num_features": 512},
#     {"name": "layer4_block2_bn2", "type": "BatchNorm2d", "num_features": 512},
#
# ]
#
# RESNET18_CIFAR10_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING = [
#     # Initial convolutional layer and batch norm
#     {'custom_name': 'conv1', 'standard_name': 'conv1.weight'},
#     {'custom_name': 'bn1', 'standard_name': 'bn1'},
#
#     # Layer 1 - Block 1
#     {'custom_name': 'layer1_block1_conv1', 'standard_name': 'layer1.0.conv1.weight'},
#     {'custom_name': 'layer1_block1_bn1', 'standard_name': 'layer1.0.bn1'},
#     {'custom_name': 'layer1_block1_conv2', 'standard_name': 'layer1.0.conv2.weight'},
#     {'custom_name': 'layer1_block1_bn2', 'standard_name': 'layer1.0.bn2'},
#
#     # Layer 1 - Block 2
#     {'custom_name': 'layer1_block2_conv1', 'standard_name': 'layer1.1.conv1.weight'},
#     {'custom_name': 'layer1_block2_bn1', 'standard_name': 'layer1.1.bn1'},
#     {'custom_name': 'layer1_block2_conv2', 'standard_name': 'layer1.1.conv2.weight'},
#     {'custom_name': 'layer1_block2_bn2', 'standard_name': 'layer1.1.bn2'},
#
#     # Layer 2 - Block 1
#     {'custom_name': 'layer2_block1_conv1', 'standard_name': 'layer2.0.conv1.weight'},
#     {'custom_name': 'layer2_block1_bn1', 'standard_name': 'layer2.0.bn1'},
#     {'custom_name': 'layer2_block1_conv2', 'standard_name': 'layer2.0.conv2.weight'},
#     {'custom_name': 'layer2_block1_bn2', 'standard_name': 'layer2.0.bn2'},
#     {'custom_name': 'layer2_block1_downsample', 'standard_name': 'layer2.0.downsample.0.weight'},
#     {'custom_name': 'layer2_block1_downsample_bn', 'standard_name': 'layer2.0.downsample.1'},
#
#     # Layer 2 - Block 2
#     {'custom_name': 'layer2_block2_conv1', 'standard_name': 'layer2.1.conv1.weight'},
#     {'custom_name': 'layer2_block2_bn1', 'standard_name': 'layer2.1.bn1'},
#     {'custom_name': 'layer2_block2_conv2', 'standard_name': 'layer2.1.conv2.weight'},
#     {'custom_name': 'layer2_block2_bn2', 'standard_name': 'layer2.1.bn2'},
#
#     # Layer 3 - Block 1
#     {'custom_name': 'layer3_block1_conv1', 'standard_name': 'layer3.0.conv1.weight'},
#     {'custom_name': 'layer3_block1_bn1', 'standard_name': 'layer3.0.bn1'},
#     {'custom_name': 'layer3_block1_conv2', 'standard_name': 'layer3.0.conv2.weight'},
#     {'custom_name': 'layer3_block1_bn2', 'standard_name': 'layer3.0.bn2'},
#     {'custom_name': 'layer3_block1_downsample', 'standard_name': 'layer3.0.downsample.0.weight'},
#     {'custom_name': 'layer3_block1_downsample_bn', 'standard_name': 'layer3.0.downsample.1'},
#
#     # Layer 3 - Block 2
#     {'custom_name': 'layer3_block2_conv1', 'standard_name': 'layer3.1.conv1.weight'},
#     {'custom_name': 'layer3_block2_bn1', 'standard_name': 'layer3.1.bn1'},
#     {'custom_name': 'layer3_block2_conv2', 'standard_name': 'layer3.1.conv2.weight'},
#     {'custom_name': 'layer3_block2_bn2', 'standard_name': 'layer3.1.bn2'},
#
#     # Layer 4 - Block 1
#     {'custom_name': 'layer4_block1_conv1', 'standard_name': 'layer4.0.conv1.weight'},
#     {'custom_name': 'layer4_block1_bn1', 'standard_name': 'layer4.0.bn1'},
#     {'custom_name': 'layer4_block1_conv2', 'standard_name': 'layer4.0.conv2.weight'},
#     {'custom_name': 'layer4_block1_bn2', 'standard_name': 'layer4.0.bn2'},
#     {'custom_name': 'layer4_block1_downsample', 'standard_name': 'layer4.0.downsample.0.weight'},
#     {'custom_name': 'layer4_block1_downsample_bn', 'standard_name': 'layer4.0.downsample.1'},
#
#     # Layer 4 - Block 2
#     {'custom_name': 'layer4_block2_conv1', 'standard_name': 'layer4.1.conv1.weight'},
#     {'custom_name': 'layer4_block2_bn1', 'standard_name': 'layer4.1.bn1'},
#     {'custom_name': 'layer4_block2_conv2', 'standard_name': 'layer4.1.conv2.weight'},
#     {'custom_name': 'layer4_block2_bn2', 'standard_name': 'layer4.1.bn2'},
#
#     # Fully connected layer
#     {'custom_name': 'fc', 'standard_name': 'fc.weight'},
# ]
#
# RESNET18_CIFAR10_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING = [
#     {'standard_name': 'conv1.weight', 'custom_name': 'conv1'},
#     {'standard_name': 'bn1', 'custom_name': 'bn1'},
#
#     # Layer 1 - Block 1
#     {'standard_name': 'layer1.0.conv1.weight', 'custom_name': 'layer1_block1_conv1'},
#     {'standard_name': 'layer1.0.bn1', 'custom_name': 'layer1_block1_bn1'},
#     {'standard_name': 'layer1.0.conv2.weight', 'custom_name': 'layer1_block1_conv2'},
#     {'standard_name': 'layer1.0.bn2', 'custom_name': 'layer1_block1_bn2'},
#
#     # Layer 1 - Block 2
#     {'standard_name': 'layer1.1.conv1.weight', 'custom_name': 'layer1_block2_conv1'},
#     {'standard_name': 'layer1.1.bn1', 'custom_name': 'layer1_block2_bn1'},
#     {'standard_name': 'layer1.1.conv2.weight', 'custom_name': 'layer1_block2_conv2'},
#     {'standard_name': 'layer1.1.bn2', 'custom_name': 'layer1_block2_bn2'},
#
#     # Layer 2 - Block 1
#     {'standard_name': 'layer2.0.conv1.weight', 'custom_name': 'layer2_block1_conv1'},
#     {'standard_name': 'layer2.0.bn1', 'custom_name': 'layer2_block1_bn1'},
#     {'standard_name': 'layer2.0.conv2.weight', 'custom_name': 'layer2_block1_conv2'},
#     {'standard_name': 'layer2.0.bn2', 'custom_name': 'layer2_block1_bn2'},
#     {'standard_name': 'layer2.0.downsample.0.weight', 'custom_name': 'layer2_block1_downsample'},
#     {'standard_name': 'layer2.0.downsample.1', 'custom_name': 'layer2_block1_downsample_bn'},
#
#     # Layer 2 - Block 2
#     {'standard_name': 'layer2.1.conv1.weight', 'custom_name': 'layer2_block2_conv1'},
#     {'standard_name': 'layer2.1.bn1', 'custom_name': 'layer2_block2_bn1'},
#     {'standard_name': 'layer2.1.conv2.weight', 'custom_name': 'layer2_block2_conv2'},
#     {'standard_name': 'layer2.1.bn2', 'custom_name': 'layer2_block2_bn2'},
#
#     # Layer 3 - Block 1
#     {'standard_name': 'layer3.0.conv1.weight', 'custom_name': 'layer3_block1_conv1'},
#     {'standard_name': 'layer3.0.bn1', 'custom_name': 'layer3_block1_bn1'},
#     {'standard_name': 'layer3.0.conv2.weight', 'custom_name': 'layer3_block1_conv2'},
#     {'standard_name': 'layer3.0.bn2', 'custom_name': 'layer3_block1_bn2'},
#     {'standard_name': 'layer3.0.downsample.0.weight', 'custom_name': 'layer3_block1_downsample'},
#     {'standard_name': 'layer3.0.downsample.1', 'custom_name': 'layer3_block1_downsample_bn'},
#
#     # Layer 3 - Block 2
#     {'standard_name': 'layer3.1.conv1.weight', 'custom_name': 'layer3_block2_conv1'},
#     {'standard_name': 'layer3.1.bn1', 'custom_name': 'layer3_block2_bn1'},
#     {'standard_name': 'layer3.1.conv2.weight', 'custom_name': 'layer3_block2_conv2'},
#     {'standard_name': 'layer3.1.bn2', 'custom_name': 'layer3_block2_bn2'},
#
#     # Layer 4 - Block 1
#     {'standard_name': 'layer4.0.conv1.weight', 'custom_name': 'layer4_block1_conv1'},
#     {'standard_name': 'layer4.0.bn1', 'custom_name': 'layer4_block1_bn1'},
#     {'standard_name': 'layer4.0.conv2.weight', 'custom_name': 'layer4_block1_conv2'},
#     {'standard_name': 'layer4.0.bn2', 'custom_name': 'layer4_block1_bn2'},
#     {'standard_name': 'layer4.0.downsample.0.weight', 'custom_name': 'layer4_block1_downsample'},
#     {'standard_name': 'layer4.0.downsample.1', 'custom_name': 'layer4_block1_downsample_bn'},
#
#     # Layer 4 - Block 2
#     {'standard_name': 'layer4.1.conv1.weight', 'custom_name': 'layer4_block2_conv1'},
#     {'standard_name': 'layer4.1.bn1', 'custom_name': 'layer4_block2_bn1'},
#     {'standard_name': 'layer4.1.conv2.weight', 'custom_name': 'layer4_block2_conv2'},
#     {'standard_name': 'layer4.1.bn2', 'custom_name': 'layer4_block2_bn2'},
#
#     # Fully connected layer
#     {'standard_name': 'fc.weight', 'custom_name': 'fc'},
# ]
