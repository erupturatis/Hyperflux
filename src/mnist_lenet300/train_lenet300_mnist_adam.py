import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import numpy as np
import torch.nn as nn
from .model_class import ModelLenet300
import wandb
from src.infrastructure.configs_layers import configs_layers_initialization_all_kaiming_sqrt5
from src.infrastructure.constants import LR_FLOW_PARAMS_ADAM, LR_FLOW_PARAMS_ADAM_RESET, get_lr_flow_params, \
    get_lr_flow_params_reset, config_adam_setup, PRUNED_MODELS_PATH
from src.infrastructure.dataset_context.dataset_context import DatasetSmallContext, DatasetSmallType, \
    dataset_context_configs_cifar10, dataset_context_configs_mnist
from src.infrastructure.layers import ConfigsNetworkMasksImportance
from src.infrastructure.others import get_device, get_model_sparsity_percent
from src.infrastructure.schedulers import PressureScheduler
from src.infrastructure.stages_context.stages_context import StagesContextPrunedTrain, StagesContextPrunedTrainArgs
from src.infrastructure.training_common import get_model_parameters_and_masks
from src.infrastructure.training_context.training_context import TrainingContextPrunedTrain, TrainingContextPrunedTrainArgs
from src.infrastructure.training_display import TrainingDisplay, ArgsTrainingDisplay
from src.infrastructure.wandb_functions import wandb_initalize, wandb_finish, Experiment, Tags
from ..common_files_experiments.train_pruned_commons import train_mixed_pruned, test_pruned


def initialize_model():
    global MODEL
    configs_network_masks = ConfigsNetworkMasksImportance(
        mask_pruning_enabled=True,
        weights_training_enabled=True,
    )
    MODEL = ModelLenet300(configs_network_masks).to(get_device())
    # MODEL.load('lenet300_mnist')

def get_epoch() -> int:
    global epoch_global
    return epoch_global

def initalize_training_display():
    global training_display
    training_display = TrainingDisplay(
        args=ArgsTrainingDisplay(
            dataset_context=dataset_context,
            average_losses_names=["Loss Data", "Loss Remaining Weights"],
            model=MODEL,
            batch_print_rate=BATCH_PRINT_RATE,
            get_epoch= get_epoch
        )
    )

def initialize_dataset_context():
    global dataset_context
    dataset_context = DatasetSmallContext(dataset=DatasetSmallType.MNIST, configs=dataset_context_configs_mnist())


def initialize_training_context():
    global training_context, MODEL

    lr_weights_training = 0.005
    lr_weights_finetuning = 0.001

    lr_weights = lr_weights_training
    lr_flow_params = get_lr_flow_params()

    weight_bias_params, flow_params, flipping_params = get_model_parameters_and_masks(MODEL)
    optimizer_weights = torch.optim.Adam(lr=lr_weights, params=weight_bias_params, weight_decay=0)
    optimizer_flow_mask = torch.optim.Adam(lr=lr_flow_params, params=flow_params, weight_decay=0)

    # reset weights are applied after pruning and before regrowth, they are the starting point for the regrowth schedulers
    training_context = TrainingContextPrunedTrain(
        TrainingContextPrunedTrainArgs(
            lr_weights_reset=lr_weights_finetuning,
            lr_flow_params_reset=get_lr_flow_params_reset(),

            l0_gamma_scaler=0,
            optimizer_weights=optimizer_weights,
            optimizer_flow_mask=optimizer_flow_mask
        )
    )

def initialize_stages_context():
    global stages_context, training_context

    pruning_end = 50
    regrowing_end = 100

    regrowth_stage_length = regrowing_end - pruning_end

    pruning_scheduler = PressureScheduler(pressure_exponent_constant=1.75, sparsity_target=0.30, epochs_target=pruning_end)
    flow_params_lr_decay_after_pruning = 0.9

    # initial learning rates are taking from optimizers, regrowth lrs are reset in the stages context before regrowing starts

    scheduler_weights_lr_during_pruning = CosineAnnealingLR(training_context.get_optimizer_weights(), T_max=pruning_end, eta_min=1e-7)
    scheduler_weights_lr_during_regrowth = CosineAnnealingLR(training_context.get_optimizer_weights(), T_max=regrowth_stage_length, eta_min=1e-7)
    scheduler_flow_params_lr_during_regrowth = LambdaLR(training_context.get_optimizer_flow_mask(), lr_lambda=lambda iter: flow_params_lr_decay_after_pruning ** iter)

    stages_context = StagesContextPrunedTrain(
        StagesContextPrunedTrainArgs(
            pruning_epoch_end=pruning_end,
            regrowth_epoch_end=regrowing_end,
            scheduler_gamma=pruning_scheduler,

            scheduler_weights_lr_during_pruning=scheduler_weights_lr_during_pruning,
            scheduler_weights_lr_during_regrowth=scheduler_weights_lr_during_regrowth,
            scheduler_flow_params_regrowth=scheduler_flow_params_lr_during_regrowth,
        ),
    )


MODEL: ModelLenet300
training_context: TrainingContextPrunedTrain
dataset_context: DatasetSmallContext
stages_context: StagesContextPrunedTrain
training_display: TrainingDisplay
epoch_global: int = 0
BATCH_PRINT_RATE = 100

def run_lenet300_mnist_adam():
    global epoch_global, MODEL
    configs_layers_initialization_all_kaiming_sqrt5()
    config_adam_setup()

    initialize_model()
    initialize_training_context()
    initialize_stages_context()
    initialize_dataset_context()
    initalize_training_display()

    wandb_initalize(
        experiment=Experiment.LENET300MNIST,
        type=Tags.TRAIN_PRUNING,
        configs=None,
        other_tags=["ADAM"]
    )

    acc = 0
    for epoch in range(1, stages_context.args.regrowth_epoch_end + 1):
        epoch_global = epoch
        dataset_context.init_data_split()
        train_mixed_pruned(
            dataset_context=dataset_context,
            training_context=training_context,
            model=MODEL,
            training_display=training_display,
        )
        acc = test_pruned(
            dataset_context=dataset_context,
            model=MODEL,
            epoch=get_epoch()
        )

        stages_context.update_context(epoch_global, get_model_sparsity_percent(MODEL))
        stages_context.step(training_context)


    MODEL.save(
        name=f"lenet300_sparsity{get_model_sparsity_percent(MODEL)}_acc{acc}",
        folder=PRUNED_MODELS_PATH
    )
    wandb_finish()
