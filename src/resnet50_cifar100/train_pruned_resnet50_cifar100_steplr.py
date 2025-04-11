import torch
from src.common_files_experiments.train_pruned_commons import train_mixed_pruned, test_pruned
from src.infrastructure.configs_layers import configs_layers_initialization_all_kaiming_sqrt5, \
    configs_layers_initialization_all_kaiming_relu
from src.infrastructure.constants import config_adam_setup, get_lr_flow_params_reset, get_lr_flow_params, \
    PRUNED_MODELS_PATH, BASELINE_RESNET18_CIFAR10, BASELINE_MODELS_PATH
from src.infrastructure.dataset_context.dataset_context import DatasetSmallContext, DatasetSmallType, dataset_context_configs_cifar100
from src.infrastructure.stages_context.stages_context import StagesContextPrunedTrain, StagesContextPrunedTrainArgs
from src.infrastructure.training_context.training_context import TrainingContextPrunedTrain, \
    TrainingContextPrunedTrainArgs
from src.infrastructure.training_display import TrainingDisplay, ArgsTrainingDisplay
from src.infrastructure.layers import ConfigsNetworkMasksImportance
from src.infrastructure.others import get_device, get_model_sparsity_percent, get_random_id
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from src.infrastructure.schedulers import PressureScheduler
from src.infrastructure.training_common import get_model_flow_params_and_weights_params
from src.infrastructure.wandb_functions import wandb_initalize, wandb_finish, Experiment, Tags
from src.resnet50_cifar100.resnet50_cifar100_class import Resnet50Cifar100

def initialize_model():
    global MODEL
    configs_network_masks = ConfigsNetworkMasksImportance(
        mask_pruning_enabled=True,
        weights_training_enabled=True,
    )
    MODEL = Resnet50Cifar100(configs_network_masks).to(get_device())
    # MODEL.load(training_configs["resume"], BASELINE_MODELS_PATH)

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
    dataset_context = DatasetSmallContext(dataset=DatasetSmallType.CIFAR100, configs=dataset_context_configs_cifar100())


def initialize_training_context():
    global training_context

    lr_weights = training_configs["weights_lr"]
    lr_flow_params = get_lr_flow_params()

    weight_params, flow_params = get_model_flow_params_and_weights_params(MODEL)
    optimizer_weights = torch.optim.SGD(lr=lr_weights, params=weight_params, momentum=0.9, weight_decay=training_configs["weight_decay"])
    optimizer_flow_mask = torch.optim.Adam(lr=lr_flow_params, params=flow_params, weight_decay=0)

    training_context = TrainingContextPrunedTrain(
        TrainingContextPrunedTrainArgs(
            l0_gamma_scaler=0,
            optimizer_weights=optimizer_weights,
            optimizer_flow_mask=optimizer_flow_mask
        )
    )

def initialize_stages_context():
    global stages_context, training_context

    pruning_end = training_configs["pruning_end"]

    pruning_scheduler = PressureScheduler(pressure_exponent_constant=1.5, sparsity_target=training_configs["target_sparsity"], epochs_target=pruning_end, step_size=0.1)

    scheduler_weights_lr = torch.optim.lr_scheduler.MultiStepLR(training_context.get_optimizer_weights(), milestones=training_configs["steps_weights_lr"], last_epoch=-1)
    scheduler_flow_params_lr = torch.optim.lr_scheduler.MultiStepLR(training_context.get_optimizer_flow_mask(), milestones=training_configs["steps_flow_lr"], last_epoch=-1)

    stages_context = StagesContextPrunedTrain(
        StagesContextPrunedTrainArgs(
            scheduler_flow_params_lr= scheduler_flow_params_lr,
            scheduler_weights_params_lr=scheduler_weights_lr,
            scheduler_gamma=pruning_scheduler,

            pruning_epoch_end=pruning_end,
        )
    )

MODEL: Resnet50Cifar100
training_context: TrainingContextPrunedTrain
dataset_context: DatasetSmallContext
stages_context: StagesContextPrunedTrain
training_display: TrainingDisplay
epoch_global: int = 0
BATCH_PRINT_RATE = 100

training_configs = {
    "epochs": 160,
    "pruning_end":120,
    "target_sparsity": 10,
    "weights_lr": 0.1,
    "weight_decay": 5e-4,
    "steps_weights_lr": [120, 140],
    "steps_flow_lr": [80, 100, 120, 140],
    "notes": '''
    Simplified setup
    '''
}

def train_resnet50_cifar100_sparse_model_steplr(sparsity_configs_aux):
    global training_configs
    global MODEL, epoch_global

    configs_layers_initialization_all_kaiming_relu()
    config_adam_setup()

    initialize_model()
    initialize_training_context()
    initialize_stages_context()
    wandb_initalize(Experiment.RESNET50CIFAR100, type=Tags.TRAIN_PRUNING, configs=training_configs, other_tags=["ADAM"], note=training_configs["notes"])
    initialize_dataset_context()
    initalize_training_display()

    for epoch in range(1, training_configs["epochs"] + 1):
        epoch_global = epoch
        dataset_context.init_data_split()

        train_mixed_pruned(
            dataset_context=dataset_context,
            training_context=training_context,
            model=MODEL,
            training_display=training_display,
        )
        test_pruned(
            dataset_context=dataset_context,
            model=MODEL,
            epoch=get_epoch()
        )

        stages_context.update_context(epoch_global, get_model_sparsity_percent(MODEL))
        stages_context.step(training_context)

    wandb_finish()
