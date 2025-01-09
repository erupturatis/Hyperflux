from enum import Enum
from typing import Tuple
import kornia.augmentation as K
from abc import ABC, abstractmethod
import torch
from src.infrastructure.others import get_device
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class TrainingContextParams:
    lr_weights: float
    lr_flow_mask: float
    l0_gamma_scaler: float

    optimizer_weights: torch.optim.Optimizer
    optimizer_flow_mask: torch.optim.Optimizer

class TrainingContext:
    def __init__(self, params: TrainingContextParams):
        self.params = params

    def get_optimizer_weights(self) -> torch.optim.Optimizer:
        return self.params.optimizer_weights

    def get_optimizer_flow_mask(self) -> torch.optim.Optimizer:
        return self.params.optimizer_flow_mask

    def set_gamma(self, gamma: float) -> None:
        self.params.l0_gamma_scaler = gamma

    def reset_param_groups_to_defaults(self) -> None:
        for param_group in self.params.optimizer_weights.param_groups:
            param_group['lr'] = self.params.lr_weights

        for param_group in self.params.optimizer_flow_mask.param_groups:
            param_group['lr'] = self.params.lr_flow_mask

        self.params.l0_gamma_scaler = 0





