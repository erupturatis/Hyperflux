from src.common_files_experiments.attributes_mutations import Mutation, mutate_attributes
from src.common_files_experiments.vanilla_attributes_resnet18 import (
    RESNET18_VANILLA_REGISTERED_LAYERS_ATTRIBUTES,
    RESNET18_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES,
    RESNET18_VANILLA_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
    RESNET18_VANILLA_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING
)
from src.infrastructure.constants import CONV2D_LAYER, FULLY_CONNECTED_LAYER

# Define the mutation for the initial convolutional layer (conv1)
replace_conv1 = Mutation(
    field_identified='name',
    value_in_field='conv1',
    action='replace',
    replacement_dict={
        "name": "conv1",
        "type": CONV2D_LAYER,
        "in_channels": 3,
        "out_channels": 64,
        "kernel_size": 3,    # Changed from 7 to 3
        "stride": 1,         # Changed from 2 to 1
        "padding": 1,        # Changed from 3 to 1
        "bias_enabled": False
    }
)

# Define the mutation for the fully connected layer (fc)
replace_fc = Mutation(
    field_identified='name',
    value_in_field='fc',
    action='replace',
    replacement_dict={
        "name": "fc",
        "type": FULLY_CONNECTED_LAYER,
        "in_features": 512,
        "out_features": 100,
        "bias_enabled": True
    }
)

# List of mutations to apply to the registered layers
cifar100_registered_mutations = [
    replace_conv1,
    replace_fc
]

# Apply mutations to the registered layers
RESNET18_CIFAR100_REGISTERED_LAYERS_ATTRIBUTES = mutate_attributes(
    attributes=RESNET18_VANILLA_REGISTERED_LAYERS_ATTRIBUTES,
    mutations=cifar100_registered_mutations
)

# Since we don't need to modify unregistered layers for CIFAR-100, keep them unchanged
RESNET18_CIFAR100_UNREGISTERED_LAYERS_ATTRIBUTES = mutate_attributes(
    attributes=RESNET18_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES,
    mutations=[]  # No mutations to apply
)

# Similarly, the layer name mappings remain unchanged
RESNET18_CIFAR100_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING = mutate_attributes(
    attributes=RESNET18_VANILLA_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
    mutations=[]  # No mutations to apply
)

RESNET18_CIFAR100_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING = mutate_attributes(
    attributes=RESNET18_VANILLA_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING,
    mutations=[]  # No mutations to apply
)
