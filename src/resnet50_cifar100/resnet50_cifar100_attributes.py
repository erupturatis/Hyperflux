from src.common_files_experiments.attributes_mutations import Mutation, mutate_attributes
from src.common_files_experiments.resnet50_vanilla_attributes import (
    RESNET50_VANILLA_REGISTERED_LAYERS_ATTRIBUTES,
    RESNET50_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES,
    RESNET50_VANILLA_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
    RESNET50_VANILLA_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING
)
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
        "kernel_size": 3,    # Changed from 7 to 3
        "stride": 1,         # Changed from 2 to 1
        "padding": 1,        # Changed from 3 to 1
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
        "in_features": 2048,  # Remains the same as ResNet50's fc layer
        "out_features": 100,  # Changed from 1000 (ImageNet) or 10 (CIFAR-10) to 100 for CIFAR-100
        "bias_enabled": True
    }
)

cifar100_registered_mutations = [
    replace_conv1,
    replace_fc
]

RESNET50_CIFAR100_REGISTERED_LAYERS_ATTRIBUTES = mutate_attributes(
    attributes=RESNET50_VANILLA_REGISTERED_LAYERS_ATTRIBUTES,
    mutations=cifar100_registered_mutations
)

RESNET50_CIFAR100_UNREGISTERED_LAYERS_ATTRIBUTES = mutate_attributes(
    attributes=RESNET50_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES,
    mutations=[]  # No mutations to apply
)

RESNET50_CIFAR100_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING = mutate_attributes(
    attributes=RESNET50_VANILLA_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
    mutations=[]  # No mutations to apply
)

RESNET50_CIFAR100_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING = mutate_attributes(
    attributes=RESNET50_VANILLA_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING,
    mutations=[]  # No mutations to apply
)
