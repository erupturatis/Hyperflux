from enum import Enum
from typing import Tuple
import kornia.augmentation as K
from abc import ABC, abstractmethod
import torch
from src.infrastructure.others import get_device
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class TrainingContextBaselineTrainArgs:
    optimizer_weights: torch.optim.Optimizer

class TrainingContextBaselineTrain:
    def __init__(self, args: TrainingContextBaselineTrainArgs):
        self.params = args

    def get_optimizer_weights(self) -> torch.optim.Optimizer:
        return self.params.optimizer_weights

@dataclass
class TrainingContextPrunedTrainArgs:
    l0_gamma_scaler: float
    optimizer_weights: torch.optim.Optimizer
    optimizer_flow_mask: torch.optim.Optimizer

class TrainingContextPrunedTrain:
    def __init__(self, params: TrainingContextPrunedTrainArgs):
        self.params = params

    def get_optimizer_weights(self) -> torch.optim.Optimizer:
        return self.params.optimizer_weights

    def get_optimizer_flow_mask(self) -> torch.optim.Optimizer:
        return self.params.optimizer_flow_mask

    def set_gamma(self, gamma: float) -> None:
        self.params.l0_gamma_scaler = gamma

@dataclass
class TrainingContextSparsityCurveArgs:
    optimizer_weights: torch.optim.Optimizer
    optimizer_flow_mask: torch.optim.Optimizer

class TrainingContextSparsityCurve:
    def __init__(self, params: TrainingContextSparsityCurveArgs):
        self.params = params

    def get_optimizer_weights(self) -> torch.optim.Optimizer:
        return self.params.optimizer_weights

    def get_optimizer_flow_mask(self) -> torch.optim.Optimizer:
        return self.params.optimizer_flow_mask