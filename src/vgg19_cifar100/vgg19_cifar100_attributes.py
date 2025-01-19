from src.common_files_experiments.attributes_mutations import Mutation, mutate_attributes
from src.common_files_experiments.vanilla_attributes_vgg19 import (
    VGG19_VANILLA_REGISTERED_LAYERS_ATTRIBUTES,
    VGG19_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES,
    VGG19_VANILLA_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
    VGG19_VANILLA_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING
)
from src.infrastructure.constants import CONV2D_LAYER, FULLY_CONNECTED_LAYER

replace_fc1_cifar100 = Mutation(
    field_identified='name',
    value_in_field='fc1',
    action='replace',
    replacement_dict={
        "name": "fc1",
        "type": FULLY_CONNECTED_LAYER,
        "in_features": 2048,
        "out_features": 4096,
        "bias_enabled": True
    }
)

replace_fc3_cifar100 = Mutation(
    field_identified='name',
    value_in_field='fc3',
    action='replace',
    replacement_dict={
        "name": "fc3",
        "type": FULLY_CONNECTED_LAYER,
        "in_features": 4096,
        "out_features": 100,
        "bias_enabled": True
    }
)

cifar100_registered_mutations = [
    replace_fc1_cifar100,
    replace_fc3_cifar100
]

VGG19_CIFAR100_REGISTERED_LAYERS_ATTRIBUTES = mutate_attributes(
    attributes=VGG19_VANILLA_REGISTERED_LAYERS_ATTRIBUTES,
    mutations=cifar100_registered_mutations
)

VGG19_CIFAR100_UNREGISTERED_LAYERS_ATTRIBUTES = mutate_attributes(
    attributes=VGG19_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES,
    mutations=[]  # No mutations to apply
)

VGG19_CIFAR100_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING = mutate_attributes(
    attributes=VGG19_VANILLA_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
    mutations=[]  # No mutations to apply
)

VGG19_CIFAR100_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING = mutate_attributes(
    attributes=VGG19_VANILLA_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING,
    mutations=[]  # No mutations to apply
)
