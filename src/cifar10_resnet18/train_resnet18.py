from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from src.config_layers import configs_layers_initialization_all_kaiming_sqrt5
from src.config_other import WANDB_REGISTER
from src.constants import WEIGHTS_ATTR, BIAS_ATTR, WEIGHTS_PRUNING_ATTR, WEIGHTS_FLIPPING_ATTR
from src.data_preprocessing import preprocess_cifar10_data_tensors_on_GPU, preprocess_cifar10_resnet_data_tensors_on_GPU
from src.layers import ConfigsNetworkMasksImportance
from src.others import get_device, ArgsDisplayModelStatistics, display_model_statistics, \
    update_args_display_model_statistics
from src.cifar10_resnet18.model_base_resnet18 import ModelBaseResnet18, ConfigsModelBaseResnet18
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
import kornia.augmentation as K
from src.schedulers import PruningScheduler
from src.training_common import get_model_parameters_and_masks
from torch.amp import GradScaler, autocast
@dataclass
class ArgsTrain:
    train_data: torch.Tensor
    train_labels: torch.Tensor

@dataclass
class TestData:
    test_data: torch.Tensor
    test_labels: torch.Tensor

@dataclass
class ArgsOptimizers:
    optimizer_weights: torch.optim
    optimizer_pruning: torch.optim
    optimizer_flipping: torch.optim

@dataclass
class ArgsOthers:
    epoch: int


def train_mixed(args_train: ArgsTrain, args_optimizers: ArgsOptimizers):
    global BATCH_SIZE, AUGMENTATIONS, MODEL, epoch_global, pruning_scheduler
    MODEL.train()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    device = get_device()

    train_data = args_train.train_data
    train_labels = args_train.train_labels
    total_data_len = len(train_data)

    optimizer_weights = args_optimizers.optimizer_weights
    optimizer_pruning = args_optimizers.optimizer_pruning
    # optimizer_flipping = args_optimizers.optimizer_flipping

    BATCH_PRINT_RATE = 100

    indices = torch.randperm(total_data_len, device=device)
    batch_indices = torch.split(indices, BATCH_SIZE)

    average_loss_names = ["Loss data", "Loss remaining weights"]
    average_loss_data = torch.tensor(0.0, device=device)
    average_loss_remaining_weights = torch.tensor(0.0, device=device)

    args_display: ArgsDisplayModelStatistics = ArgsDisplayModelStatistics(
        BATCH_PRINT_RATE=BATCH_PRINT_RATE,
        DATA_LENGTH=total_data_len,
        batch_size=BATCH_SIZE,
        average_loss_names=average_loss_names,
        model=MODEL
    )

    total, remaining = MODEL.get_parameters_pruning_statistics()
    pruning_scheduler.record_state(remaining)
    pruning_scheduler.step()

    scaler = GradScaler('cuda')  # Initialize GradScaler for mixed precision

    for batch_idx, batch in enumerate(batch_indices):
        data = train_data[batch].to(device, non_blocking=True)
        target = train_labels[batch].to(device, non_blocking=True)
        data = AUGMENTATIONS(data)

        # Zero the gradients for all optimizers
        optimizer_weights.zero_grad()
        optimizer_pruning.zero_grad()
        # optimizer_flipping.zero_grad()

        with autocast('cuda'):
            output = MODEL(data)
            loss_remaining_weights = MODEL.get_remaining_parameters_loss() * pruning_scheduler.get_multiplier()   # Ensure this is float
            loss_data = criterion(output, target)
            loss = loss_remaining_weights + loss_data

        # Scale the loss and call backward()
        scaler.scale(loss).backward()

        # Optionally, accumulate average loss for monitoring
        average_loss_data += loss_data.detach()
        average_loss_remaining_weights += 0

        # Step the optimizers using the scaled gradients
        scaler.step(optimizer_weights)
        scaler.step(optimizer_pruning)
        # scaler.step(optimizer_flipping)

        # Update the scaler for the next iteration
        scaler.update()

        # Print and reset average losses at specified intervals
        if (batch_idx + 1) % BATCH_PRINT_RATE == 0 or (batch_idx + 1) * BATCH_SIZE >= total_data_len:
            update_args_display_model_statistics(
                args_display,
                [average_loss_data, average_loss_remaining_weights],
                batch_idx,
                epoch_global
            )
            display_model_statistics(args_display)
            average_loss_data = torch.tensor(0.0, device=device)
            average_loss_remaining_weights = torch.tensor(0.0, device=device)

    # Optionally, return any metrics or state if needed



def train(args_train: ArgsTrain, args_optimizers: ArgsOptimizers):
    global BATCH_SIZE, AUGMENTATIONS, MODEL, epoch_global, pruning_scheduler
    MODEL.train()

    criterion = nn.CrossEntropyLoss(label_smoothing= 0.1)
    device = get_device()

    train_data = args_train.train_data
    train_labels = args_train.train_labels
    total_data_len = len(train_data)

    optimizer_weights = args_optimizers.optimizer_weights
    # optimizer_pruning = args_optimizers.optimizer_pruning
    # optimizer_flipping = args_optimizers.optimizer_flipping

    BATCH_PRINT_RATE = 100

    indices = torch.randperm(total_data_len, device=device)
    batch_indices = torch.split(indices, BATCH_SIZE)

    average_loss_names = ["Loss data", "Loss remaining weights"]
    average_loss_data = torch.tensor(0.0).to(device)
    average_loss_remaining_weights = torch.tensor(0.0).to(device)

    args_display: ArgsDisplayModelStatistics = ArgsDisplayModelStatistics(
        BATCH_PRINT_RATE=BATCH_PRINT_RATE,
        DATA_LENGTH=total_data_len,
        batch_size=BATCH_SIZE,
        average_loss_names=average_loss_names,
        model=MODEL
    )

    total, remaining = MODEL.get_parameters_pruning_statistics()
    # pruning_scheduler.record_state(remaining)
    # pruning_scheduler.step()

    for batch_idx, batch in enumerate(batch_indices):
        data = train_data[batch]
        target = train_labels[batch]
        data = AUGMENTATIONS(data)

        optimizer_weights.zero_grad()
        # optimizer_pruning.zero_grad()
        # optimizer_flipping.zero_grad()

        output = MODEL(data)
        # loss_remaining_weights = MODEL.get_remaining_parameters_loss() * pruning_scheduler.get_multiplier() * 0
        loss_remaining_weights = 0

        loss_data = criterion(output, target)
        average_loss_data += loss_data
        average_loss_remaining_weights += loss_remaining_weights

        loss = loss_remaining_weights + loss_data

        if (batch_idx + 1) % BATCH_PRINT_RATE == 0 or (batch_idx + 1) == total_data_len:
            update_args_display_model_statistics(args_display, [average_loss_data, average_loss_remaining_weights], batch_idx, epoch_global)
            display_model_statistics(args_display)
            average_loss_data = torch.tensor(0.0).to(device)
            average_loss_remaining_weights = torch.tensor(0.0).to(device)

        loss.backward()

        optimizer_weights.step()
        # optimizer_pruning.step()
        # optimizer_flipping.step()


def test(args_test: TestData):
    global BATCH_SIZE, MODEL, epoch_global

    MODEL.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    test_data = args_test.test_data
    test_labels = args_test.test_labels

    test_loss = 0
    correct = 0

    total_data_len = len(test_data)
    with torch.no_grad():
        batch_indices = torch.split(torch.arange(total_data_len, device=get_device()), BATCH_SIZE)

        for batch in batch_indices:
            data = test_data[batch]
            target = test_labels[batch]

            output = MODEL(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total_data_len
    accuracy = 100.0 * correct / total_data_len

    total, remaining = MODEL.get_parameters_pruning_statistics()
    remain_percent = remaining / total * 100

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{total_data_len} ({accuracy:.0f}%)\n",
        f"Remaining parameters: {remain_percent:.2f}%"
    )

    if WANDB_REGISTER:
        wandb.log({"epoch": epoch_global, "test_loss": test_loss, "accuracy": accuracy, "remaining_parameters": remain_percent})

    return accuracy  # Return accuracy for custom table


BATCH_SIZE = 128
MODEL: ModelBaseResnet18
AUGMENTATIONS = nn.Sequential(
    K.RandomCrop((32, 32), padding=4),
    K.RandomRotation(degrees=10.0),
    K.RandomHorizontalFlip(p=0.5),
).to(get_device())
pruning_scheduler: PruningScheduler
epoch_global: int = 0

def run_cifar10_resnet18():
    configs_layers_initialization_all_kaiming_sqrt5()
    global MODEL, BATCH_SIZE, epoch_global, pruning_scheduler
    # fine tuning and from scratch
    lr_weight_bias = 0.0001
    # lr_weight_bias = 0.1

    lr_custom_params = 0.001
    stop_epoch = 400
    num_epochs = 600

    configs_network_masks = ConfigsNetworkMasksImportance(
        mask_pruning_enabled=True,
        mask_flipping_enabled=False,
        weights_training_enabled=True,
    )
    configs_model_base_resnet18 = ConfigsModelBaseResnet18(num_classes=10)
    MODEL = ModelBaseResnet18(configs_model_base_resnet18, configs_network_masks).to(get_device())
    # MODEL.load_pretrained_pytorch()
    MODEL.load('/data/pretrained/resnet18-cifar10-trained95')
    pruning_scheduler = PruningScheduler(exponent_constant=2, pruning_target=0.005, epochs_target=stop_epoch, total_parameters=MODEL.get_parameters_total_count())
    train_data, train_labels, test_data, test_labels = preprocess_cifar10_resnet_data_tensors_on_GPU()

    if WANDB_REGISTER:
        wandb.init(
            project="Dump",
            config={
                "batch_size": BATCH_SIZE,
                "num_epochs": num_epochs,
                "lr_weight_bias": lr_weight_bias,
                "lr_custom_params": lr_custom_params,
            },
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    # Initialize custom ResNet model
    weight_bias_params, pruning_params, flipping_params = get_model_parameters_and_masks(MODEL)

    optimizer_weights = torch.optim.SGD(lr=lr_weight_bias, params= weight_bias_params, momentum=0.9, weight_decay=1e-4)
    optimizer_pruning = torch.optim.AdamW(pruning_params, lr=lr_custom_params)
    optimizer_flipping = torch.optim.AdamW(flipping_params, lr=lr_custom_params)

    scheduler_decay_after_pruning = 0.9

    scheduler_regrowing_weights = CosineAnnealingLR(optimizer_weights, T_max=(num_epochs - stop_epoch))
    scheduler_pruning = LambdaLR(optimizer_pruning, lr_lambda=lambda iter: scheduler_decay_after_pruning ** iter)

    for epoch in range(1, num_epochs + 1):
        epoch_global = epoch
        train_mixed(ArgsTrain(train_data, train_labels), ArgsOptimizers(optimizer_weights, optimizer_pruning, optimizer_flipping))
        test(TestData(test_data, test_labels))

        if epoch > stop_epoch:
            scheduler_regrowing_weights.step()
            scheduler_pruning.step()

    MODEL.save("/data/pretrained/resnet18-cifar10-pruned")

    print("Training complete")
    # wandb.finish()
