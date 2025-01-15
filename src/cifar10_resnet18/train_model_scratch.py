from dataclasses import dataclass
import torch
import torch.nn as nn
from src.infrastructure.configs_layers import configs_layers_initialization_all_kaiming_sqrt5
from src.infrastructure.constants import LR_FLOW_PARAMS_ADAM, config_adam_setup, LR_FLOW_PARAMS_RESET, \
    get_lr_flow_params_reset, get_lr_flow_params
from src.infrastructure.dataset_context.dataset_context import DatasetSmallContext, DatasetSmallType, dataset_context_configs_cifar10
from src.infrastructure.stages_context.stages_context import StagesContextPrunedTrain, StagesContextPrunedTrainArgs, StagesContextBaselineTrain, \
    StagesContextBaselineTrainArgs
from src.infrastructure.training_context.training_context import TrainingContextPrunedTrain, TrainingContextSparsityCurveArgs, \
    TrainingContextBaselineTrain, TrainingContextBaselineTrainArgs
from src.infrastructure.training_display import TrainingDisplay, ArgsTrainingDisplay
from src.infrastructure.layers import ConfigsNetworkMasksImportance
from src.infrastructure.others import get_device, get_model_sparsity_percent, get_random_id
from src.cifar10_resnet18.model_class import ModelBaseResnet18, ConfigsModelBaseResnet18
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from src.infrastructure.schedulers import PressureScheduler, PressureScheduler
from src.infrastructure.training_common import get_model_parameters
from torch.amp import GradScaler, autocast
from src.infrastructure.wandb_functions import wandb_initalize, wandb_finish, Experiment, Tags, \
    wandb_snapshot_baseline


def train_mixed():
    global MODEL, epoch_global,  dataset_context, training_display, training_context

    MODEL.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer_weights = training_context.get_optimizer_weights()

    scaler = GradScaler('cuda')
    while dataset_context.any_data_training_available():
        data, target = dataset_context.get_training_data_and_labels()

        optimizer_weights.zero_grad()

        with autocast('cuda'):
            output = MODEL(data)
            loss_data = criterion(output, target)

        scaler.scale(loss_data).backward()
        scaler.step(optimizer_weights)
        scaler.update()

        training_display.record_losses([loss_data.item()])

def test():
    global MODEL, epoch_global, dataset_context

    MODEL.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    test_loss = 0
    correct = 0

    with torch.no_grad():
        while dataset_context.any_data_testing_available():
            data, target = dataset_context.get_testing_data_and_labels()

            output = MODEL(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    total_data_len = dataset_context.get_data_testing_length()
    test_loss /= total_data_len
    accuracy = 100.0 * correct / total_data_len

    remain_percent = get_model_sparsity_percent(MODEL)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{total_data_len} ({accuracy:.0f}%)"
    )
    wandb_snapshot_baseline(
        epoch=get_epoch(),
        accuracy=accuracy,
        test_loss=test_loss,
    )
    return accuracy


def initialize_model():
    global MODEL
    configs_network_masks = ConfigsNetworkMasksImportance(
        mask_pruning_enabled=False,
        weights_training_enabled=True,
    )
    configs_model_base_resnet18 = ConfigsModelBaseResnet18(num_classes=10)
    MODEL = ModelBaseResnet18(configs_model_base_resnet18, configs_network_masks).to(get_device())

def get_epoch() -> int:
    global epoch_global
    return epoch_global

def initalize_training_display():
    global training_display
    training_display = TrainingDisplay(
        args=ArgsTrainingDisplay(
            dataset_context=dataset_context,
            average_losses_names=["Loss Data"],
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

    lr = 0.1
    weight_bias_params = get_model_parameters(MODEL)
    optimizer_weights = torch.optim.SGD(lr=lr, params= weight_bias_params, momentum=0.9, weight_decay=5e-4)

    training_context = TrainingContextBaselineTrain(
        TrainingContextBaselineTrainArgs(
            optimizer_weights=optimizer_weights,
        )
    )

def initialize_stages_context():
    global stages_context, training_context

    training_end = 3
    scheduler_weights_lr_during_training = CosineAnnealingLR(training_context.get_optimizer_weights(), T_max=training_end, eta_min=1e-7)

    stages_context = StagesContextBaselineTrain(
        StagesContextBaselineTrainArgs(
            training_end=training_end,
            scheduler_weights_lr_during_training=scheduler_weights_lr_during_training,
        )
    )

MODEL: ModelBaseResnet18
training_context: TrainingContextBaselineTrain
dataset_context: DatasetSmallContext
stages_context: StagesContextBaselineTrain
training_display: TrainingDisplay
epoch_global: int = 0
BATCH_PRINT_RATE = 100

sparsity_configs = {
    "pruning_end": 400,
    "regrowing_end": 600,
    "target_sparsity": 0.5,
    "lr_flow_params_decay_regrowing": 0.95
}

def train_cifar10_resnet18_from_scratch():
    global MODEL, epoch_global
    configs_layers_initialization_all_kaiming_sqrt5()

    initialize_model()
    initialize_training_context()
    initialize_stages_context()
    wandb_initalize(experiment=Experiment.RESNET18CIFAR10, type=Tags.BASELINE, configs=sparsity_configs)
    initialize_dataset_context()
    initalize_training_display()

    acc = 0
    for epoch in range(1, stages_context.args.training_end + 1):
        epoch_global = epoch
        dataset_context.init_data_split()
        train_mixed()
        acc = test()

        stages_context.update_context(epoch_global)
        stages_context.step(training_context)

    MODEL.save(f"/network_baselines/resnet18_cifar10_accuracy{acc}%_{get_random_id()}")

    print("Training complete")
    wandb_finish()
