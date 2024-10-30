import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from src.common import ConfigsNetworkMasks, LayerLinear, MaskPruningFunction, MaskFlipFunction
from src.utils import get_device
from src.constants import MASK_PRUNING_ATTR, MASK_FLIPPING_ATTR


class ModelMnistFNN(nn.Module):
    def __init__(self, config_network_mask: ConfigsNetworkMasks):
        super(ModelMnistFNN, self).__init__()
        self.fc1 = LayerLinear(
            configs_linear={'in_features': 28*28, 'out_features': 300},
            configs_network=config_network_mask
        )
        self.fc2 = LayerLinear(
            configs_linear={'in_features': 300, 'out_features': 100},
            configs_network=config_network_mask
        )
        self.fc3 = LayerLinear(
            configs_linear={'in_features': 100, 'out_features': 10},
            configs_network=config_network_mask
        )

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_masked_loss(self) -> torch.Tensor:
        total = 0
        masked = torch.tensor(0, device=get_device(), dtype=torch.float)
        for layer in [self.fc1, self.fc2, self.fc3]:
            total += layer.weights.numel()
            mask = torch.sigmoid(getattr(layer, MASK_PRUNING_ATTR))
            # Apply threshold at 0.5 to get binary mask
            masked += mask.sum()

        return masked / total

    def get_masked_percentage(self) -> float:
        total = 0
        masked = torch.tensor(0, device=get_device(), dtype=torch.float)
        for layer in [self.fc1, self.fc2, self.fc3]:
            total += layer.weights.numel()
            mask = torch.sigmoid(getattr(layer, MASK_PRUNING_ATTR))
            # Apply threshold at 0.5 to get binary mask
            mask_thresholded = (mask >= 0.5).float()
            masked += mask_thresholded.sum()

        return masked.item() / total

    def get_flipped_percentage(self) -> float:
        total = 0
        flipped = torch.tensor(0, device=get_device(), dtype=torch.float)
        for layer in [self.fc1, self.fc2, self.fc3]:
            total += layer.weights.numel()
            mask = torch.sigmoid(getattr(layer, MASK_FLIPPING_ATTR))
            # Apply threshold at 0.5 to get binary mask
            mask_thresholded = (mask >= 0.5).float()
            flipped += mask_thresholded.sum()

        return (total-flipped.item()) / total

    # other methods
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


