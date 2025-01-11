from dataclasses import dataclass
import torch
import torch.nn as nn
from src.infrastructure.configs_layers import configs_layers_initialization_all_kaiming_sqrt5
from src.infrastructure.constants import LR_FLOW_PARAMS_ADAM
from src.infrastructure.dataset_context.dataset_context import DatasetSmallContext, DatasetSmallType, dataset_context_configs_cifar10
from src.infrastructure.stages_context.stages_context import StagesContext, StagesContextArgs
from src.infrastructure.training_context.training_context import TrainingContext, TrainingContextParams
from src.infrastructure.training_display import TrainingDisplay, ArgsTrainingDisplay
from src.infrastructure.layers import ConfigsNetworkMasksImportance
from src.infrastructure.others import get_device, get_model_sparsity_percent
from src.cifar10_resnet18.model_class import ModelBaseResnet18, ConfigsModelBaseResnet18
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from src.infrastructure.schedulers import PressureScheduler, PressureScheduler
from src.infrastructure.training_common import get_model_parameters_and_masks
from torch.amp import GradScaler, autocast
from src.infrastructure.wandb_functions import wandb_initalize, wandb_finish


def train_mixed():
    global MODEL, epoch_global,  dataset_context, training_display, training_context
    MODEL.train()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer_weights = training_context.get_optimizer_weights()
    optimizer_pruning = training_context.get_optimizer_flow_mask()

    scaler = GradScaler('cuda')

    while dataset_context.any_data_training_available():
        data, target = dataset_context.get_training_data_and_labels()

        optimizer_weights.zero_grad()
        optimizer_pruning.zero_grad()

        with autocast('cuda'):
            output = MODEL(data)
            loss_remaining_weights = MODEL.get_remaining_parameters_loss() * training_context.params.l0_gamma_scaler
            loss_data = criterion(output, target)
            loss = loss_remaining_weights + loss_data

        scaler.scale(loss).backward()
        scaler.step(optimizer_weights)
        scaler.step(optimizer_pruning)
        scaler.update()

        training_display.record_losses([loss_data.item(), loss_remaining_weights.item()], training_context)


# def train(args_train: ArgsTrain, args_optimizers: ArgsOptimizers):
#     global BATCH_SIZE, AUGMENTATIONS, MODEL, epoch_global, pruning_scheduler
#     MODEL.train()
#
#     criterion = nn.CrossEntropyLoss(label_smoothing= 0.1)
#     device = get_device()
#
#     train_data = args_train.train_data
#     train_labels = args_train.train_labels
#     total_data_len = len(train_data)
#
#     optimizer_weights = args_optimizers.optimizer_weights
#     # optimizer_pruning = args_optimizers.optimizer_pruning
#     # optimizer_flipping = args_optimizers.optimizer_flipping
#
#     BATCH_PRINT_RATE = 100
#
#     indices = torch.randperm(total_data_len, device=device)
#     batch_indices = torch.split(indices, BATCH_SIZE)
#
#     average_loss_names = ["Loss data", "Loss remaining weights"]
#     average_loss_data = torch.tensor(0.0).to(device)
#     average_loss_remaining_weights = torch.tensor(0.0).to(device)
#
#     args_display: ArgsDisplayModelStatistics = ArgsDisplayModelStatistics(
#         BATCH_PRINT_RATE=BATCH_PRINT_RATE,
#         DATA_LENGTH=total_data_len,
#         batch_size=BATCH_SIZE,
#         average_loss_names=average_loss_names,
#         model=MODEL
#     )
#
#     total, remaining = MODEL.get_parameters_pruning_statistics()
#     # pruning_scheduler.record_state(remaining)
#     # pruning_scheduler.step()
#
#     for batch_idx, batch in enumerate(batch_indices):
#         data = train_data[batch]
#         target = train_labels[batch]
#         data = AUGMENTATIONS(data)
#
#         optimizer_weights.zero_grad()
#         # optimizer_pruning.zero_grad()
#         # optimizer_flipping.zero_grad()
#
#         output = MODEL(data)
#         # loss_remaining_weights = MODEL.get_remaining_parameters_loss() * pruning_scheduler.get_multiplier() * 0
#         loss_remaining_weights = 0
#
#         loss_data = criterion(output, target)
#         average_loss_data += loss_data
#         average_loss_remaining_weights += loss_remaining_weights
#
#         loss = loss_remaining_weights + loss_data
#
#         if (batch_idx + 1) % BATCH_PRINT_RATE == 0 or (batch_idx + 1) == total_data_len:
#             update_args_display_model_statistics(args_display, [average_loss_data, average_loss_remaining_weights], batch_idx, epoch_global)
#             display_model_statistics(args_display)
#             average_loss_data = torch.tensor(0.0).to(device)
#             average_loss_remaining_weights = torch.tensor(0.0).to(device)
#
#         loss.backward()
#
#         optimizer_weights.step()
#         # optimizer_pruning.step()
#         # optimizer_flipping.step()


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
    print(
        f"Remaining parameters: {remain_percent:.2f}%"
    )


def initialize_model():
    global MODEL
    configs_network_masks = ConfigsNetworkMasksImportance(
        mask_pruning_enabled=True,
        mask_flipping_enabled=False,
        weights_training_enabled=True,
    )
    configs_model_base_resnet18 = ConfigsModelBaseResnet18(num_classes=10)
    MODEL = ModelBaseResnet18(configs_model_base_resnet18, configs_network_masks).to(get_device())
    MODEL.load('resnet18-cifar10-trained95')

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

    lr_weights_finetuning = 0.0001
    lr_flow_params = LR_FLOW_PARAMS_ADAM

    weight_bias_params, pruning_params, flipping_params = get_model_parameters_and_masks(MODEL)
    optimizer_weights = torch.optim.SGD(lr=lr_weights_finetuning, params= weight_bias_params, momentum=0.9, weight_decay=1e-4)
    optimizer_flow_mask = torch.optim.AdamW(pruning_params, lr=lr_flow_params)

    training_context = TrainingContext(
        TrainingContextParams(
            lr_weights=lr_weights_finetuning,
            lr_flow_params=LR_FLOW_PARAMS_ADAM,
            l0_gamma_scaler=0,
            optimizer_weights=optimizer_weights,
            optimizer_flow_mask=optimizer_flow_mask
        )
    )

def initialize_stages_context():
    global stages_context, training_context

    pruning_end = 400
    regrowing_end = 600

    regrowth_stage_length = regrowing_end - pruning_end
    pruning_scheduler = PressureScheduler(pressure_exponent_constant=1.75, sparsity_target=0.5, epochs_target=pruning_end)

    scheduler_decay_after_pruning = 0.9
    scheduler_regrowing_weights = CosineAnnealingLR(training_context.get_optimizer_weights(), T_max=regrowth_stage_length, eta_min=1e-7)
    scheduler_flow_params = LambdaLR(training_context.get_optimizer_flow_mask(), lr_lambda=lambda iter: scheduler_decay_after_pruning ** iter)

    stages_context = StagesContext(
        StagesContextArgs(
            pruning_epoch_end=pruning_end,
            regrowth_epoch_end=regrowing_end,
            scheduler_gamma=pruning_scheduler,

            scheduler_weights_lr_during_pruning=None,
            scheduler_flow_params_regrowth=scheduler_flow_params,
            scheduler_weights_lr_during_regrowth=scheduler_regrowing_weights,
        )
    )

MODEL: ModelBaseResnet18
training_context: TrainingContext
dataset_context: DatasetSmallContext
stages_context: StagesContext
training_display: TrainingDisplay
epoch_global: int = 0
BATCH_PRINT_RATE = 100

def run_cifar10_resnet18():
    configs_layers_initialization_all_kaiming_sqrt5()
    global MODEL, epoch_global,  dataset_context, training_display, training_context, stages_context

    initialize_model()
    initialize_training_context()
    initialize_stages_context()
    wandb_initalize()
    initialize_dataset_context()
    initalize_training_display()

    for epoch in range(1, stages_context.args.regrowth_epoch_end + 1):
        epoch_global = epoch
        dataset_context.init_data_split()
        train_mixed()
        test()

        stages_context.update_context(epoch_global, get_model_sparsity_percent(MODEL))
        stages_context.step(training_context)

    MODEL.save("/data/pretrained/resnet18-cifar10-pruned")

    print("Training complete")
    wandb_finish()
