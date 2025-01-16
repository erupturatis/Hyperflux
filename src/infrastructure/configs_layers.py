from src.infrastructure.layer_initializations import kaiming_sqrt5, kaiming_sqrt0, bad_initialization
from src.infrastructure.mask_functions import MaskPruningFunctionSigmoid, MaskFlipFunctionSigmoid
from src.infrastructure.parameters_mask_processors import get_parameters_pruning_statistics_step_, \
    get_parameters_flipped_statistics_sigmoid_, get_parameters_pruning_statistics_vanilla_, \
    get_parameters_pruning_step_aproximation_constant_
from typing import Dict

MaskPruningFunction = MaskPruningFunctionSigmoid
MaskFlipFunction = MaskFlipFunctionSigmoid

get_parameters_pruning = get_parameters_pruning_step_aproximation_constant_
get_parameters_pruning_statistics = get_parameters_pruning_statistics_step_

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