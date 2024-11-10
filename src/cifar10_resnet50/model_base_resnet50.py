from typing import List
import torch
import torch.nn as nn
from dataclasses import dataclass
from src.others import get_device, prefix_path_with_root
from src.layers import (
    LayerConv2,
    ConfigsNetworkMasks,
    LayerLinear,
    LayerComposite,
    LayerPrimitive,
    get_layers_primitive,
    get_remaining_parameters_loss,
    get_layer_composite_pruning_statistics,
    ConfigsLayerConv2,
    ConfigsLayerLinear,
    get_layer_composite_flipped_statistics,
    get_parameters_total_count,
)

@dataclass
class ConfigsModelBaseResnet50:
    num_classes: int

class ModelBaseResnet50(LayerComposite):
    def __init__(self, configs_model_base_resnet: ConfigsModelBaseResnet50, configs_network_masks: ConfigsNetworkMasks):
        super(ModelBaseResnet50, self).__init__()
        self.registered_layers = []
        self.relu = nn.ReLU(inplace=True)
        self.NUM_OUTPUT_CLASSES = configs_model_base_resnet.num_classes

        # Initial convolutional layer with bias=False
        self.conv1 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.conv1)
        self.bn1 = nn.BatchNorm2d(64)
        #self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ############################################
        # Layer 1 (Conv2_x): 3 Bottleneck Blocks
        ############################################

        ############################################
        # Block 1   
        ############################################
        # Conv1 
        self.layer1_block1_conv1 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer1_block1_conv1)
        self.layer1_block1_bn1 = nn.BatchNorm2d(64)

        # Conv2
        self.layer1_block1_conv2 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer1_block1_conv2)
        self.layer1_block1_bn2 = nn.BatchNorm2d(64)

        # Conv3
        self.layer1_block1_conv3 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=64,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer1_block1_conv3)
        self.layer1_block1_bn3 = nn.BatchNorm2d(256)

        # Downsample
        self.layer1_block1_downsample = LayerConv2(
            ConfigsLayerConv2(
                in_channels=64,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer1_block1_downsample)
        self.layer1_block1_downsample_bn = nn.BatchNorm2d(256)

        ############################################
        # Block 2
        ############################################
        # Conv1
        self.layer1_block2_conv1 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=256,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer1_block2_conv1)
        self.layer1_block2_bn1 = nn.BatchNorm2d(64)

        # Conv2
        self.layer1_block2_conv2 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer1_block2_conv2)
        self.layer1_block2_bn2 = nn.BatchNorm2d(64)

        # Conv3
        self.layer1_block2_conv3 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=64,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer1_block2_conv3)
        self.layer1_block2_bn3 = nn.BatchNorm2d(256)

        ############################################
        # Block 3
        ############################################
        # Conv1
        self.layer1_block3_conv1 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=256,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer1_block3_conv1)
        self.layer1_block3_bn1 = nn.BatchNorm2d(64)

        # Conv2
        self.layer1_block3_conv2 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer1_block3_conv2)
        self.layer1_block3_bn2 = nn.BatchNorm2d(64)

        # Conv3
        self.layer1_block3_conv3 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=64,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer1_block3_conv3)
        self.layer1_block3_bn3 = nn.BatchNorm2d(256)

        ############################################
        # Layer 2 (Conv3_x): 4 Bottleneck Blocks
        ############################################

        ############################################
        # Block 1 with Downsampling
        ############################################
        # Conv1
        self.layer2_block1_conv1 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=256,
                out_channels=128,
                kernel_size=1,
                stride=2,  # Stride of 2 for downsampling
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer2_block1_conv1)
        self.layer2_block1_bn1 = nn.BatchNorm2d(128)

        # Conv2
        self.layer2_block1_conv2 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer2_block1_conv2)
        self.layer2_block1_bn2 = nn.BatchNorm2d(128)

        # Conv3
        self.layer2_block1_conv3 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=128,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer2_block1_conv3)
        self.layer2_block1_bn3 = nn.BatchNorm2d(512)

        # Downsample
        self.layer2_block1_downsample = LayerConv2(
            ConfigsLayerConv2(
                in_channels=256,
                out_channels=512,
                kernel_size=1,
                stride=2,  # Stride of 2 for downsampling
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer2_block1_downsample)
        self.layer2_block1_downsample_bn = nn.BatchNorm2d(512)

        ############################################
        # Block 2
        ############################################
        self.layer2_block2_conv1 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=512,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer2_block2_conv1)
        self.layer2_block2_bn1 = nn.BatchNorm2d(128)

        self.layer2_block2_conv2 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer2_block2_conv2)
        self.layer2_block2_bn2 = nn.BatchNorm2d(128)

        self.layer2_block2_conv3 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=128,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer2_block2_conv3)
        self.layer2_block2_bn3 = nn.BatchNorm2d(512)

        ############################################
        # Block 3
        ############################################
        self.layer2_block3_conv1 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=512,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer2_block3_conv1)
        self.layer2_block3_bn1 = nn.BatchNorm2d(128)

        self.layer2_block3_conv2 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer2_block3_conv2)
        self.layer2_block3_bn2 = nn.BatchNorm2d(128)

        self.layer2_block3_conv3 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=128,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer2_block3_conv3)
        self.layer2_block3_bn3 = nn.BatchNorm2d(512)

        ############################################
        # Block 4
        ############################################
        self.layer2_block4_conv1 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=512,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer2_block4_conv1)
        self.layer2_block4_bn1 = nn.BatchNorm2d(128)

        self.layer2_block4_conv2 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer2_block4_conv2)
        self.layer2_block4_bn2 = nn.BatchNorm2d(128)

        self.layer2_block4_conv3 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=128,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer2_block4_conv3)
        self.layer2_block4_bn3 = nn.BatchNorm2d(512)

        ############################################
        # Layer 3 (Conv4_x): 6 Bottleneck Blocks
        ############################################

        ############################################
        # Block 1 with Downsampling
        ############################################
        self.layer3_block1_conv1 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=512,
                out_channels=256,
                kernel_size=1,
                stride=2,  # Stride of 2 for downsampling
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer3_block1_conv1)
        self.layer3_block1_bn1 = nn.BatchNorm2d(256)

        self.layer3_block1_conv2 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer3_block1_conv2)
        self.layer3_block1_bn2 = nn.BatchNorm2d(256)

        self.layer3_block1_conv3 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=256,
                out_channels=1024,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer3_block1_conv3)
        self.layer3_block1_bn3 = nn.BatchNorm2d(1024)

        # Downsample
        self.layer3_block1_downsample = LayerConv2(
            ConfigsLayerConv2(
                in_channels=512,
                out_channels=1024,
                kernel_size=1,
                stride=2,  # Stride of 2 for downsampling
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer3_block1_downsample)
        self.layer3_block1_downsample_bn = nn.BatchNorm2d(1024)

        ############################################
        # Blocks 2 to 6
        ############################################
        for i in range(2, 7):
            # Conv1
            setattr(self, f'layer3_block{i}_conv1', LayerConv2(
                ConfigsLayerConv2(
                    in_channels=1024,
                    out_channels=256,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias_enabled=False
                ),
                configs_network_masks
            ))
            self.registered_layers.append(getattr(self, f'layer3_block{i}_conv1'))
            setattr(self, f'layer3_block{i}_bn1', nn.BatchNorm2d(256))

            # Conv2
            setattr(self, f'layer3_block{i}_conv2', LayerConv2(
                ConfigsLayerConv2(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias_enabled=False
                ),
                configs_network_masks
            ))
            self.registered_layers.append(getattr(self, f'layer3_block{i}_conv2'))
            setattr(self, f'layer3_block{i}_bn2', nn.BatchNorm2d(256))

            # Conv3
            setattr(self, f'layer3_block{i}_conv3', LayerConv2(
                ConfigsLayerConv2(
                    in_channels=256,
                    out_channels=1024,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias_enabled=False
                ),
                configs_network_masks
            ))
            self.registered_layers.append(getattr(self, f'layer3_block{i}_conv3'))
            setattr(self, f'layer3_block{i}_bn3', nn.BatchNorm2d(1024))

        ############################################
        # Layer 4 (Conv5_x): 3 Bottleneck Blocks
        ############################################

        ############################################
        # Block 1 with Downsampling
        ############################################
        self.layer4_block1_conv1 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=1024,
                out_channels=512,
                kernel_size=1,
                stride=2,  # Stride of 2 for downsampling
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer4_block1_conv1)
        self.layer4_block1_bn1 = nn.BatchNorm2d(512)

        self.layer4_block1_conv2 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer4_block1_conv2)
        self.layer4_block1_bn2 = nn.BatchNorm2d(512)

        self.layer4_block1_conv3 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=512,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer4_block1_conv3)
        self.layer4_block1_bn3 = nn.BatchNorm2d(2048)

        # Downsample
        self.layer4_block1_downsample = LayerConv2(
            ConfigsLayerConv2(
                in_channels=1024,
                out_channels=2048,
                kernel_size=1,
                stride=2,  # Stride of 2 for downsampling
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer4_block1_downsample)
        self.layer4_block1_downsample_bn = nn.BatchNorm2d(2048)

        ############################################
        # Block 2
        ############################################
        self.layer4_block2_conv1 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=2048,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer4_block2_conv1)
        self.layer4_block2_bn1 = nn.BatchNorm2d(512)

        self.layer4_block2_conv2 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer4_block2_conv2)
        self.layer4_block2_bn2 = nn.BatchNorm2d(512)

        self.layer4_block2_conv3 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=512,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer4_block2_conv3)
        self.layer4_block2_bn3 = nn.BatchNorm2d(2048)

        ############################################
        # Block 3
        ############################################
        self.layer4_block3_conv1 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=2048,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer4_block3_conv1)
        self.layer4_block3_bn1 = nn.BatchNorm2d(512)

        self.layer4_block3_conv2 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer4_block3_conv2)
        self.layer4_block3_bn2 = nn.BatchNorm2d(512)

        self.layer4_block3_conv3 = LayerConv2(
            ConfigsLayerConv2(
                in_channels=512,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_enabled=False
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.layer4_block3_conv3)
        self.layer4_block3_bn3 = nn.BatchNorm2d(2048)

        ############################################
        # Final Layers
        ############################################
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = LayerLinear(
            ConfigsLayerLinear(
                in_features=2048,
                out_features=self.NUM_OUTPUT_CLASSES
            ),
            configs_network_masks
        )
        self.registered_layers.append(self.fc)

        self.load_pretrained_weights()

    def get_remaining_parameters_loss(self) -> torch.Tensor:
        total, sigmoid = get_remaining_parameters_loss(self)
        return sigmoid / total

    def get_layers_primitive(self) -> List[LayerPrimitive]:
        return get_layers_primitive(self)

    def get_parameters_pruning_statistics(self) -> any:
        return get_layer_composite_pruning_statistics(self)

    def get_parameters_flipped_statistics(self) -> any:
        return get_layer_composite_flipped_statistics(self)

    def get_parameters_total_count(self) -> int:
        total = get_parameters_total_count(self)
        return total

    def load_pretrained_weights(self):
        import torchvision.models as models

        # Load the pretrained ResNet-50 model from torchvision
        pretrained_model = models.resnet50(pretrained=True)

        pretrained_state = pretrained_model.state_dict()
        own_state = self.state_dict()

        # Handle the initial conv1 layer separately due to different kernel sizes
        # Pretrained conv1 weight shape: (64, 3, 7, 7)
        # Our conv1 weight shape: (64, 3, 3, 3)
        pretrained_conv1_weight = pretrained_state['conv1.weight']

        # Extract the central 3x3 portion of the 7x7 kernels
        conv1_weight = pretrained_conv1_weight[:, :, 2:5, 2:5]
        own_state['conv1.weights'].copy_(conv1_weight)

        # Copy batch normalization parameters for the initial layer
        own_state['bn1.weight'].copy_(pretrained_state['bn1.weight'])
        own_state['bn1.bias'].copy_(pretrained_state['bn1.bias'])
        own_state['bn1.running_mean'].copy_(pretrained_state['bn1.running_mean'])
        own_state['bn1.running_var'].copy_(pretrained_state['bn1.running_var'])

        # Now map and copy the rest of the layers
        # Define the number of blocks in each layer
        layers_blocks = [3, 4, 6, 3]  # For ResNet-50

        for layer_num in range(1, 5):
            num_blocks = layers_blocks[layer_num - 1]
            for block_num in range(num_blocks):
                # Pretrained model uses zero-based indexing for blocks
                pretrained_block_prefix = f'layer{layer_num}.{block_num}'
                # Our model uses one-based indexing for blocks
                own_block_prefix = f'layer{layer_num}_block{block_num + 1}'

                # Copy conv and bn layers within each block
                for conv_num in range(1, 4):  # conv1, conv2, conv3
                    # Convolutional layers
                    pretrained_conv_name = f'{pretrained_block_prefix}.conv{conv_num}.weight'
                    own_conv_name = f'{own_block_prefix}_conv{conv_num}.weights'

                    if pretrained_conv_name in pretrained_state and own_conv_name in own_state:
                        own_state[own_conv_name].copy_(pretrained_state[pretrained_conv_name])
                    else:
                        print(f"Skipping '{pretrained_conv_name}' or '{own_conv_name}' due to missing keys or mismatch.")

                    # BatchNorm layers
                    pretrained_bn_prefix = f'{pretrained_block_prefix}.bn{conv_num}'
                    own_bn_prefix = f'{own_block_prefix}_bn{conv_num}'

                    for bn_param in ['weight', 'bias', 'running_mean', 'running_var']:
                        pretrained_bn_name = f'{pretrained_bn_prefix}.{bn_param}'
                        own_bn_name = f'{own_bn_prefix}.{bn_param}'

                        if pretrained_bn_name in pretrained_state and own_bn_name in own_state:
                            own_state[own_bn_name].copy_(pretrained_state[pretrained_bn_name])
                        else:
                            print(f"Skipping '{pretrained_bn_name}' or '{own_bn_name}' due to missing keys or mismatch.")

                # Handle downsample layers if present
                downsample_conv_name = f'{pretrained_block_prefix}.downsample.0.weight'
                if downsample_conv_name in pretrained_state:
                    own_downsample_conv_name = f'{own_block_prefix}_downsample.weights'

                    if own_downsample_conv_name in own_state:
                        own_state[own_downsample_conv_name].copy_(pretrained_state[downsample_conv_name])
                    else:
                        print(f"Skipping '{own_downsample_conv_name}' due to missing key or mismatch.")

                    # Downsample BatchNorm layers
                    pretrained_downsample_bn_prefix = f'{pretrained_block_prefix}.downsample.1'
                    own_downsample_bn_prefix = f'{own_block_prefix}_downsample_bn'

                    for bn_param in ['weight', 'bias', 'running_mean', 'running_var']:
                        pretrained_downsample_bn_name = f'{pretrained_downsample_bn_prefix}.{bn_param}'
                        own_downsample_bn_name = f'{own_downsample_bn_prefix}.{bn_param}'

                        if pretrained_downsample_bn_name in pretrained_state and own_downsample_bn_name in own_state:
                            own_state[own_downsample_bn_name].copy_(pretrained_state[pretrained_downsample_bn_name])
                        else:
                            print(f"Skipping '{pretrained_downsample_bn_name}' or '{own_downsample_bn_name}' due to missing keys or mismatch.")

        # Skip loading weights for the final fully connected layer
        # since the number of classes is different (1000 vs. 10)
        print("Successfully loaded pretrained weights where possible.")

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool1(x)

        ############################################
        # Layer 1
        ############################################

        ############################################
        # Block 1
        ############################################
        identity = self.layer1_block1_downsample_bn(self.layer1_block1_downsample(x))

        out = self.layer1_block1_conv1(x)
        out = self.layer1_block1_bn1(out)
        out = self.relu(out)

        out = self.layer1_block1_conv2(out)
        out = self.layer1_block1_bn2(out)
        out = self.relu(out)

        out = self.layer1_block1_conv3(out)
        out = self.layer1_block1_bn3(out)

        out += identity
        out = self.relu(out)

        ############################################
        # Block 2
        ############################################
        identity = out

        out = self.layer1_block2_conv1(out)
        out = self.layer1_block2_bn1(out)
        out = self.relu(out)

        out = self.layer1_block2_conv2(out)
        out = self.layer1_block2_bn2(out)
        out = self.relu(out)

        out = self.layer1_block2_conv3(out)
        out = self.layer1_block2_bn3(out)

        out += identity
        out = self.relu(out)

        ############################################
        # Block 3
        ############################################
        identity = out

        out = self.layer1_block3_conv1(out)
        out = self.layer1_block3_bn1(out)
        out = self.relu(out)

        out = self.layer1_block3_conv2(out)
        out = self.layer1_block3_bn2(out)
        out = self.relu(out)

        out = self.layer1_block3_conv3(out)
        out = self.layer1_block3_bn3(out)

        out += identity
        out = self.relu(out)

        ############################################
        # Layer 2
        ############################################

        ############################################
        # Block 1 with Downsampling
        ############################################
        identity = self.layer2_block1_downsample_bn(self.layer2_block1_downsample(out))

        out = self.layer2_block1_conv1(out)
        out = self.layer2_block1_bn1(out)
        out = self.relu(out)

        out = self.layer2_block1_conv2(out)
        out = self.layer2_block1_bn2(out)
        out = self.relu(out)

        out = self.layer2_block1_conv3(out)
        out = self.layer2_block1_bn3(out)

        out += identity
        out = self.relu(out)

        ############################################
        # Block 2
        ############################################
        identity = out

        out = self.layer2_block2_conv1(out)
        out = self.layer2_block2_bn1(out)
        out = self.relu(out)

        out = self.layer2_block2_conv2(out)
        out = self.layer2_block2_bn2(out)
        out = self.relu(out)

        out = self.layer2_block2_conv3(out)
        out = self.layer2_block2_bn3(out)

        out += identity
        out = self.relu(out)

        ############################################
        # Block 3
        ############################################
        identity = out

        out = self.layer2_block3_conv1(out)
        out = self.layer2_block3_bn1(out)
        out = self.relu(out)

        out = self.layer2_block3_conv2(out)
        out = self.layer2_block3_bn2(out)
        out = self.relu(out)

        out = self.layer2_block3_conv3(out)
        out = self.layer2_block3_bn3(out)

        out += identity
        out = self.relu(out)

        ############################################
        # Block 4
        ############################################
        identity = out

        out = self.layer2_block4_conv1(out)
        out = self.layer2_block4_bn1(out)
        out = self.relu(out)

        out = self.layer2_block4_conv2(out)
        out = self.layer2_block4_bn2(out)
        out = self.relu(out)

        out = self.layer2_block4_conv3(out)
        out = self.layer2_block4_bn3(out)

        out += identity
        out = self.relu(out)

        ############################################
        # Layer 3
        ############################################

        ############################################
        # Block 1 with Downsampling
        ############################################
        identity = self.layer3_block1_downsample_bn(self.layer3_block1_downsample(out))

        out = self.layer3_block1_conv1(out)
        out = self.layer3_block1_bn1(out)
        out = self.relu(out)

        out = self.layer3_block1_conv2(out)
        out = self.layer3_block1_bn2(out)
        out = self.relu(out)

        out = self.layer3_block1_conv3(out)
        out = self.layer3_block1_bn3(out)

        out += identity
        out = self.relu(out)

        ############################################
        # Blocks 2 to 6
        ############################################
        for i in range(2, 7):
            identity = out

            out = getattr(self, f'layer3_block{i}_conv1')(out)
            out = getattr(self, f'layer3_block{i}_bn1')(out)
            out = self.relu(out)

            out = getattr(self, f'layer3_block{i}_conv2')(out)
            out = getattr(self, f'layer3_block{i}_bn2')(out)
            out = self.relu(out)

            out = getattr(self, f'layer3_block{i}_conv3')(out)
            out = getattr(self, f'layer3_block{i}_bn3')(out)

            out += identity
            out = self.relu(out)

        ############################################
        # Layer 4
        ############################################

        ############################################
        # Block 1 with Downsampling
        ############################################
        identity = self.layer4_block1_downsample_bn(self.layer4_block1_downsample(out))

        out = self.layer4_block1_conv1(out)
        out = self.layer4_block1_bn1(out)
        out = self.relu(out)

        out = self.layer4_block1_conv2(out)
        out = self.layer4_block1_bn2(out)
        out = self.relu(out)

        out = self.layer4_block1_conv3(out)
        out = self.layer4_block1_bn3(out)

        out += identity
        out = self.relu(out)

        ############################################
        # Block 2
        ############################################
        identity = out

        out = self.layer4_block2_conv1(out)
        out = self.layer4_block2_bn1(out)
        out = self.relu(out)

        out = self.layer4_block2_conv2(out)
        out = self.layer4_block2_bn2(out)
        out = self.relu(out)

        out = self.layer4_block2_conv3(out)
        out = self.layer4_block2_bn3(out)

        out += identity
        out = self.relu(out)

        ############################################
        # Block 3
        ############################################
        identity = out

        out = self.layer4_block3_conv1(out)
        out = self.layer4_block3_bn1(out)
        out = self.relu(out)

        out = self.layer4_block3_conv2(out)
        out = self.layer4_block3_bn2(out)
        out = self.relu(out)

        out = self.layer4_block3_conv3(out)
        out = self.layer4_block3_bn3(out)

        out += identity
        out = self.relu(out)

        ############################################
        # Final Layers
        ############################################
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
