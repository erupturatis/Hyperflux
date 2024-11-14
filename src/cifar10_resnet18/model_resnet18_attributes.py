# Registered layers
RESNET18_CIFAR10_REGISTERED_LAYERS_ATTRIBUTES = [
    # Initial convolutional layer
    {"name": "conv1", "type": "LayerConv2", "in_channels": 3, "out_channels": 64,
     "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},

    # Layer 1 - Block 1
    {"name": "layer1_block1_conv1", "type": "LayerConv2", "in_channels": 64,
     "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
    {"name": "layer1_block1_conv2", "type": "LayerConv2", "in_channels": 64,
     "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},

    # Layer 1 - Block 2
    {"name": "layer1_block2_conv1", "type": "LayerConv2", "in_channels": 64,
     "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
    {"name": "layer1_block2_conv2", "type": "LayerConv2", "in_channels": 64,
     "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},

    # Layer 2 - Block 1
    {"name": "layer2_block1_conv1", "type": "LayerConv2", "in_channels": 64,
     "out_channels": 128, "kernel_size": 3, "stride": 2, "padding": 1, "bias_enabled": False},
    {"name": "layer2_block1_conv2", "type": "LayerConv2", "in_channels": 128,
     "out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
    {"name": "layer2_block1_downsample", "type": "LayerConv2", "in_channels": 64,
     "out_channels": 128, "kernel_size": 1, "stride": 2, "padding": 0, "bias_enabled": False},

    # Layer 2 - Block 2
    {"name": "layer2_block2_conv1", "type": "LayerConv2", "in_channels": 128,
     "out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
    {"name": "layer2_block2_conv2", "type": "LayerConv2", "in_channels": 128,
     "out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},

    # Layer 3 - Block 1
    {"name": "layer3_block1_conv1", "type": "LayerConv2", "in_channels": 128,
     "out_channels": 256, "kernel_size": 3, "stride": 2, "padding": 1, "bias_enabled": False},
    {"name": "layer3_block1_conv2", "type": "LayerConv2", "in_channels": 256,
     "out_channels": 256, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
    {"name": "layer3_block1_downsample", "type": "LayerConv2", "in_channels": 128,
     "out_channels": 256, "kernel_size": 1, "stride": 2, "padding": 0, "bias_enabled": False},

    # Layer 3 - Block 2
    {"name": "layer3_block2_conv1", "type": "LayerConv2", "in_channels": 256,
     "out_channels": 256, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
    {"name": "layer3_block2_conv2", "type": "LayerConv2", "in_channels": 256,
     "out_channels": 256, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},

    # Layer 4 - Block 1
    {"name": "layer4_block1_conv1", "type": "LayerConv2", "in_channels": 256,
     "out_channels": 512, "kernel_size": 3, "stride": 2, "padding": 1, "bias_enabled": False},
    {"name": "layer4_block1_conv2", "type": "LayerConv2", "in_channels": 512,
     "out_channels": 512, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
    {"name": "layer4_block1_downsample", "type": "LayerConv2", "in_channels": 256,
     "out_channels": 512, "kernel_size": 1, "stride": 2, "padding": 0, "bias_enabled": False},

    # Layer 4 - Block 2
    {"name": "layer4_block2_conv1", "type": "LayerConv2", "in_channels": 512,
     "out_channels": 512, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
    {"name": "layer4_block2_conv2", "type": "LayerConv2", "in_channels": 512,
     "out_channels": 512, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},

    # Fully connected layer
    {"name": "fc", "type": "LayerLinear", "in_features": 512, "out_features": 10}
]

# Unregistered layers (e.g., batch norms, activations, pooling)
RESNET18_CIFAR10_UNREGISTERED_LAYERS_ATTRIBUTES = [
    # Initial batch normalization
    {"name": "bn1", "type": "BatchNorm2d", "num_features": 64},

    # Layer 1 - Block 1
    {"name": "layer1_block1_bn1", "type": "BatchNorm2d", "num_features": 64},
    {"name": "layer1_block1_bn2", "type": "BatchNorm2d", "num_features": 64},

    # Layer 1 - Block 2
    {"name": "layer1_block2_bn1", "type": "BatchNorm2d", "num_features": 64},
    {"name": "layer1_block2_bn2", "type": "BatchNorm2d", "num_features": 64},

    # Layer 2 - Block 1
    {"name": "layer2_block1_bn1", "type": "BatchNorm2d", "num_features": 128},
    {"name": "layer2_block1_bn2", "type": "BatchNorm2d", "num_features": 128},
    {"name": "layer2_block1_downsample_bn", "type": "BatchNorm2d", "num_features": 128},

    # Layer 2 - Block 2
    {"name": "layer2_block2_bn1", "type": "BatchNorm2d", "num_features": 128},
    {"name": "layer2_block2_bn2", "type": "BatchNorm2d", "num_features": 128},

    # Layer 3 - Block 1
    {"name": "layer3_block1_bn1", "type": "BatchNorm2d", "num_features": 256},
    {"name": "layer3_block1_bn2", "type": "BatchNorm2d", "num_features": 256},
    {"name": "layer3_block1_downsample_bn", "type": "BatchNorm2d", "num_features": 256},

    # Layer 3 - Block 2
    {"name": "layer3_block2_bn1", "type": "BatchNorm2d", "num_features": 256},
    {"name": "layer3_block2_bn2", "type": "BatchNorm2d", "num_features": 256},

    # Layer 4 - Block 1
    {"name": "layer4_block1_bn1", "type": "BatchNorm2d", "num_features": 512},
    {"name": "layer4_block1_bn2", "type": "BatchNorm2d", "num_features": 512},
    {"name": "layer4_block1_downsample_bn", "type": "BatchNorm2d", "num_features": 512},

    # Layer 4 - Block 2
    {"name": "layer4_block2_bn1", "type": "BatchNorm2d", "num_features": 512},
    {"name": "layer4_block2_bn2", "type": "BatchNorm2d", "num_features": 512},

    # Activation functions and pooling layers
    {"name": "avgpool", "type": "AdaptiveAvgPool2d", "output_size": (1, 1)}
]
