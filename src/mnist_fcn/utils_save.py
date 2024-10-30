import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from src.layers import MaskPruningFunction, MaskFlipFunction

def get_layerwise_weights(self):
    weights = {}
    for i, layer in enumerate([self.fc1, self.fc2, self.fc3]):
        weight = layer.weight.detach().cpu().numpy()
        weights[f'fc{i}_weights'] = np.where(weight < 0, -1, 1)
    return weights

def get_layerwise_masks(self):
    masks = {}
    for i, layer in enumerate([self.fc1, self.fc2, self.fc3]):
        if layer.mask_enabled:
            binary_mask = MaskPruningFunction.apply(layer.mask_param).detach().cpu().numpy()
            print(binary_mask)
        if layer.signs_enabled:
            masks[f'fc{i}_weights'] = MaskFlipFunction.apply(layer.signs_mask_param).detach().cpu().numpy()
    return masks

def save_weights(self):
    # Apply masks to weights before saving
    masked_fc1_weight = self.fc1.weight * MaskPruningFunction.apply(self.fc1.mask_param)
    masked_fc2_weight = self.fc2.weight * MaskPruningFunction.apply(self.fc2.mask_param)
    masked_fc3_weight = self.fc3.weight * MaskPruningFunction.apply(self.fc3.mask_param)

    if self.fc1.signs_enabled:
        masked_fc1_weight = masked_fc1_weight * MaskFlipFunction.apply(self.fc1.signs_mask_param)
    if self.fc2.signs_enabled:
        masked_fc2_weight = masked_fc2_weight * MaskFlipFunction.apply(self.fc2.signs_mask_param)
    if self.fc3.signs_enabled:
        masked_fc3_weight = masked_fc3_weight * MaskFlipFunction.apply(self.fc3.signs_mask_param)

    # Save the masked weights
    weights_to_save = {
        'fc1.weight': masked_fc1_weight.detach().cpu(),
        'fc2.weight': masked_fc2_weight.detach().cpu(),
        'fc3.weight': masked_fc3_weight.detach().cpu()
    }

    if self.fc1.bias is not None:
        weights_to_save['fc1.bias'] = self.fc1.bias.detach().cpu()
    if self.fc2.bias is not None:
        weights_to_save['fc2.bias'] = self.fc2.bias.detach().cpu()
    if self.fc3.bias is not None:
        weights_to_save['fc3.bias'] = self.fc3.bias.detach().cpu()

    torch.save(weights_to_save, r"XAI_paper\nn_weights\model_v1_with_mask.pth")
    print(f"Weights with masks applied saved!")
