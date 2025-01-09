from typing import TYPE_CHECKING

import torch

from src.infrastructure.constants import WEIGHTS_ATTR, BIAS_ATTR, WEIGHTS_PRUNING_ATTR, WEIGHTS_FLIPPING_ATTR
if TYPE_CHECKING:
    from src.layers import LayerComposite


def get_model_parameters_and_masks(model: 'LayerComposite') -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    weight_bias_params = []
    flipping_params = []
    pruning_params = []

    for name, param in model.named_parameters():
        if WEIGHTS_ATTR in name or BIAS_ATTR in name:
            weight_bias_params.append(param)
        if WEIGHTS_PRUNING_ATTR in name:
            pruning_params.append(param)
        if WEIGHTS_FLIPPING_ATTR in name:
            flipping_params.append(param)


    return weight_bias_params, pruning_params, flipping_params
