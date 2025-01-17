from src.common_files_experiments.attributes_mutations import Mutation, mutate_attributes
from src.common_files_experiments.vanilla_attributes_vgg19 import (
    VGG19_VANILLA_REGISTERED_LAYERS_ATTRIBUTES,
    VGG19_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES,
    VGG19_VANILLA_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
    VGG19_VANILLA_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING
)
from src.infrastructure.constants import CONV2D_LAYER, FULLY_CONNECTED_LAYER

# Define the mutation for the first fully connected layer (fc1) for CIFAR-100
replace_fc1_cifar100 = Mutation(
    field_identified='name',
    value_in_field='fc1',
    action='replace',
    replacement_dict={
        "name": "fc1",
        "type": FULLY_CONNECTED_LAYER,
        "in_features": 512,    # Changed from 25088 to 512
        "out_features": 4096,
        "bias_enabled": True
    }
)

# Define the mutation for the third fully connected layer (fc3) for CIFAR-100
replace_fc3_cifar100 = Mutation(
    field_identified='name',
    value_in_field='fc3',
    action='replace',
    replacement_dict={
        "name": "fc3",
        "type": FULLY_CONNECTED_LAYER,
        "in_features": 4096,   # Remains the same
        "out_features": 100,   # Changed from 1000 to 100
        "bias_enabled": True
    }
)

# List of mutations to apply for CIFAR-100
cifar100_registered_mutations = [
    replace_fc1_cifar100,
    replace_fc3_cifar100
]

# Apply mutations to the registered layers for CIFAR-100
VGG19_CIFAR100_REGISTERED_LAYERS_ATTRIBUTES = mutate_attributes(
    attributes=VGG19_VANILLA_REGISTERED_LAYERS_ATTRIBUTES,
    mutations=cifar100_registered_mutations
)

# Unregistered layers remain unchanged for CIFAR-100
VGG19_CIFAR100_UNREGISTERED_LAYERS_ATTRIBUTES = mutate_attributes(
    attributes=VGG19_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES,
    mutations=[]  # No mutations to apply
)

# Layer name mappings remain unchanged for CIFAR-100
VGG19_CIFAR100_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING = mutate_attributes(
    attributes=VGG19_VANILLA_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
    mutations=[]  # No mutations to apply
)

VGG19_CIFAR100_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING = mutate_attributes(
    attributes=VGG19_VANILLA_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING,
    mutations=[]  # No mutations to apply
)
