import torch
from src.common_files_experiments.train_pruned_commons import train_mixed_pruned, test_pruned
from src.infrastructure.configs_layers import configs_layers_initialization_all_kaiming_sqrt5, \
    configs_layers_initialization_all_kaiming_relu
from src.infrastructure.constants import config_adam_setup, get_lr_flow_params_reset, get_lr_flow_params, \
    PRUNED_MODELS_PATH, BASELINE_RESNET18_CIFAR10, BASELINE_MODELS_PATH, BASELINE_RESNET50_CIFAR10
from src.infrastructure.dataset_context.dataset_context import DatasetSmallContext, DatasetSmallType, dataset_context_configs_cifar10
from src.infrastructure.stages_context.stages_context import StagesContextPrunedTrain, StagesContextPrunedTrainArgs
from src.infrastructure.training_context.training_context import TrainingContextPrunedTrain, \
    TrainingContextPrunedTrainArgs
from src.infrastructure.training_display import TrainingDisplay, ArgsTrainingDisplay
from src.infrastructure.layers import ConfigsNetworkMasksImportance
from src.infrastructure.others import get_device, get_model_sparsity_percent
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from src.infrastructure.schedulers import PressureScheduler
from src.infrastructure.training_common import get_model_parameters_and_masks
from src.infrastructure.wandb_functions import wandb_initalize, wandb_finish, Experiment, Tags
from src.resnet50_cifar10.resnet50_cifar10_class import Resnet50Cifar10
from src.resnet18_cifar10.resnet18_cifar10_class import Resnet18Cifar10


def initialize_model():
    global MODEL
    configs_network_masks = ConfigsNetworkMasksImportance(
        mask_pruning_enabled=True,
        weights_training_enabled=True,
    )
    MODEL = Resnet50Cifar10(configs_network_masks).to(get_device())
    MODEL.load("resnet50_cifar10_accuracy94.64%", BASELINE_MODELS_PATH)

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
    dataset_context = DatasetSmallContext(dataset=DatasetSmallType.CIFAR10, configs=dataset_context_configs_cifar10())


def initialize_training_context():
    global training_context

    lr_weights_finetuning = 0.001
    lr_flow_params = get_lr_flow_params()

    weight_bias_params, flow_params, _ = get_model_parameters_and_masks(MODEL)
    optimizer_weights = torch.optim.SGD(lr=lr_weights_finetuning, params= weight_bias_params, momentum=0.9, weight_decay=0)
    optimizer_flow_mask = torch.optim.Adam(lr=lr_flow_params, params=flow_params, weight_decay=0)

    training_context = TrainingContextPrunedTrain(
        TrainingContextPrunedTrainArgs(
            lr_weights_reset=lr_weights_finetuning,
            lr_flow_params_reset=get_lr_flow_params(),
            l0_gamma_scaler=0,
            optimizer_weights=optimizer_weights,
            optimizer_flow_mask=optimizer_flow_mask
        )
    )

def initialize_stages_context():
    global stages_context, training_context

    pruning_end = sparsity_configs["pruning_end"]
    regrowing_end = sparsity_configs["regrowing_end"]
    regrowth_stage_length = regrowing_end - pruning_end

    pruning_scheduler = PressureScheduler(pressure_exponent_constant=1.5, sparsity_target=sparsity_configs["target_sparsity"], epochs_target=pruning_end)
    scheduler_decay_after_pruning = sparsity_configs["lr_flow_params_decay_regrowing"]

    scheduler_weights_lr_during_pruning = LambdaLR(training_context.get_optimizer_weights(), lr_lambda=lambda step: 1 ** step)
    scheduler_weights_lr_during_regrowth = CosineAnnealingLR(training_context.get_optimizer_weights(), T_max=regrowth_stage_length, eta_min=1e-6)
    scheduler_flow_params_lr_during_regrowth = LambdaLR(training_context.get_optimizer_flow_mask(), lr_lambda=lambda iter: scheduler_decay_after_pruning ** iter if iter < 50 else 0)

    stages_context = StagesContextPrunedTrain(
        StagesContextPrunedTrainArgs(
            pruning_epoch_end=pruning_end,
            regrowth_epoch_end=regrowing_end,
            scheduler_gamma=pruning_scheduler,

            scheduler_weights_lr_during_pruning=scheduler_weights_lr_during_pruning,
            scheduler_flow_params_regrowth=scheduler_flow_params_lr_during_regrowth,
            scheduler_weights_lr_during_regrowth=scheduler_weights_lr_during_regrowth,
        )
    )

MODEL: Resnet50Cifar10
training_context: TrainingContextPrunedTrain
dataset_context: DatasetSmallContext
stages_context: StagesContextPrunedTrain
training_display: TrainingDisplay
epoch_global: int = 0
BATCH_PRINT_RATE = 100

sparsity_configs = {
    "pruning_end": 125,
    "regrowing_end": 200,
    "target_sparsity": 0.21,
    "lr_flow_params_decay_regrowing": 0.9
}

def train_resnet50_cifar10_sparse_model():
    global MODEL, epoch_global

    configs_layers_initialization_all_kaiming_relu()
    config_adam_setup()

    initialize_model()
    initialize_training_context()
    initialize_stages_context()
    wandb_initalize(Experiment.RESNET50CIFAR10, type=Tags.TRAIN_PRUNING, configs=sparsity_configs,other_tags=["ADAM"])
    initialize_dataset_context()
    initalize_training_display()

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
        name= f"resnet50_cifar10_sparsity{get_model_sparsity_percent(MODEL)}_acc{acc}",
        folder= PRUNED_MODELS_PATH
    )
    print("Training complete")
    wandb_finish()
