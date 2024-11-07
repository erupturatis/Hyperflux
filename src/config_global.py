from src.layer_initializations import kaiming_sqrt5, kaiming_sqrt0, bad_initialization
from src.mask_functions import MaskPruningFunctionSigmoid, MaskFlipFunctionSigmoid, MaskPruningFunctionLeaky, \
    MaskFlipFunctionLeaky, MaskPruningFunctionLinear
from src.parameters_mask_processors import get_parameters_pruning_leaky, get_parameters_pruning_statistics_leaky, \
    get_parameters_flipped_statistics_leaky, get_parameters_pruning_sigmoid, get_parameters_pruning_statistics_sigmoid, \
    get_parameters_flipped_statistics_sigmoid
from typing import Dict, Union

# MaskPruningFunction = MaskPruningFunctionLeaky
# MaskFlipFunction = MaskFlipFunctionLeaky
# get_parameters_pruning = get_parameters_pruning_leaky
# get_parameters_pruning_statistics = get_parameters_pruning_statistics_leaky
# get_parameters_flipped_statistics = get_parameters_flipped_statistics_leaky

# MaskPruningFunction = MaskPruningFunctionLeaky
# MaskFlipFunction = MaskFlipFunctionLeaky
# get_parameters_pruning = get_parameters_pruning_leaky
# get_parameters_pruning_statistics = get_parameters_pruning_statistics_leaky
# get_parameters_flipped_statistics = get_parameters_flipped_statistics_leaky

MaskPruningFunction = MaskPruningFunctionSigmoid
MaskFlipFunction = MaskFlipFunctionSigmoid
get_parameters_pruning = get_parameters_pruning_sigmoid
get_parameters_pruning_statistics = get_parameters_pruning_statistics_sigmoid
get_parameters_flipped_statistics = get_parameters_flipped_statistics_sigmoid

_configs_layers_init = {
    "fcn": kaiming_sqrt5,
    "conv2d": kaiming_sqrt5
}

def configs_get_layers_all_initialization(layer_name: str) -> Dict:
    return _configs_layers_init

def configs_get_layers_initialization(layer_name) -> callable:
    return _configs_layers_init[layer_name]

def configs_layers_initialization_all_kaiming_sqrt5():
    global _configs_layers_init
    _configs_layers_init = {
        "fcn": kaiming_sqrt5,
        "conv2d": kaiming_sqrt5
    }

def configs_layers_initialization_all_kaiming_sqrt0():
    global _configs_layers_init
    _configs_layers_init = {
        "fcn": kaiming_sqrt0,
        "conv2d": kaiming_sqrt0
    }

def configs_layers_initialization_all_bad():
    global _configs_layers_init
    _configs_layers_init = {
        "fcn": bad_initialization,
        "conv2d": bad_initialization
    }