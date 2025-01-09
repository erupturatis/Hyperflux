from enum import Enum
import warnings
from typing import Tuple
import kornia.augmentation as K
from abc import ABC, abstractmethod
import torch
from src.infrastructure.others import get_device
import torch.nn as nn
from dataclasses import dataclass

from src.infrastructure.schedulers import PressureScheduler
from src.infrastructure.training_context.training_context import TrainingContext


@dataclass
class StagesContextArgs:
    scheduler_weights_pruning: torch.optim.lr_scheduler.LRScheduler

    scheduler_weights_regrowth: torch.optim.lr_scheduler.LRScheduler
    scheduler_flow_params_regrowth: torch.optim.lr_scheduler.LRScheduler

    scheduler_gamma: PressureScheduler

    pruning_epoch_end: int
    regrowth_epoch_end: int

class StagesContext:
    def __init__(self, args: StagesContextArgs):
        self.epoch = 1
        self.sparsity_percent = 100
        self.args = args

        if self.args.scheduler_weights_pruning == None:
            warnings.warn("Scheduler weights pruning is disabled")
        if self.args.scheduler_weights_regrowth == None:
            warnings.warn("Scheduler weights regrowth is disabled")
        if self.args.scheduler_flow_params_regrowth == None:
            warnings.warn("Scheduler flow params regrowth is disabled")
        if self.args.scheduler_gamma == None:
            warnings.warn("Scheduler gamma is disabled")

    def update_context(self, epoch: int, sparsity_percent: float):
        self.epoch = epoch
        self.sparsity_percent = sparsity_percent

    def step(self, training_context: TrainingContext):
        print("Start epoch lr", training_context.params.optimizer_weights.param_groups[0]['lr'])
        print("Start epoch lr2", training_context.params.optimizer_flow_mask.param_groups[0]['lr'])

        if self.epoch <= self.args.pruning_epoch_end:
            # run weights scheduler
            if self.args.scheduler_weights_pruning is not None:
                self.args.scheduler_weights_pruning.step()

            # run gamma scheduler
            self.args.scheduler_gamma.step(self.epoch, self.sparsity_percent)
            gamma = self.args.scheduler_gamma.get_multiplier()
            training_context.set_gamma(gamma)

        if self.epoch == self.args.pruning_epoch_end + 1:
            # here regrowing just starts, we reset param groups
            training_context.reset_param_groups_to_defaults()

        if self.epoch >= self.args.pruning_epoch_end + 1 and self.epoch <= self.args.regrowth_epoch_end:
            if self.args.scheduler_weights_regrowth is not None:
                self.args.scheduler_weights_regrowth.step()

            # we decay flow params lr, to stop flipping and limit regrowing
            if self.args.scheduler_flow_params_regrowth is not None:
                self.args.scheduler_flow_params_regrowth.step()

            # we remove pressure
            training_context.set_gamma(0)

        print("End epoch lr", training_context.params.optimizer_weights.param_groups[0]['lr'])
        print("End epoch lr2", training_context.params.optimizer_flow_mask.param_groups[0]['lr'])




