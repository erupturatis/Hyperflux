import math
import torch
from matplotlib.pyplot import connect
from src.config_layers import configs_layers_initialization_all_kaiming_sqrt0, configs_get_layers_initialization, \
    configs_layers_initialization_all_bad, configs_layers_initialization_all_kaiming_sqrt5
from src.constants import WEIGHTS_ATTR, BIAS_ATTR, WEIGHTS_PRUNING_ATTR, WEIGHTS_FLIPPING_ATTR
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.data_preprocessing import preprocess_cifar10_data_loaders, preprocess_cifar10_data_tensors_on_GPU
from src.layers import ConfigsNetworkMasks
from src.others import get_device, ArgsDisplayModelStatistics, update_args_display_model_statistics, \
    display_model_statistics
from src.cifar10_conv2.model_conv2 import ModelCifar10Conv2
import numpy as np
from torch.optim.lr_scheduler import StepLR

from src.schedulers import PruningScheduler
from src.training_common import get_model_parameters_and_masks
import wandb
from dataclasses import dataclass

@dataclass
class ArgsTrain:
    train_data: torch.Tensor
    train_labels: torch.Tensor

@dataclass
class TestData:
    test_data: torch.Tensor
    test_labels: torch.Tensor


@dataclass
class ArgsTrain:
    train_data: torch.Tensor
    train_labels: torch.Tensor

@dataclass
class ArgsOptimizers:
    optimizer: torch.optim

def train(args_train: ArgsTrain, args_optimizers: ArgsOptimizers):
    global MODEL, epoch_global, BATCH_SIZE, pruning_scheduler
    MODEL.train()
    criterion = nn.CrossEntropyLoss()
    device = get_device()

    BATCH_PRINT_RATE = 100

    train_data = args_train.train_data
    train_labels = args_train.train_labels

    total_data_len = len(train_data)
    indices = torch.randperm(total_data_len, device=device)
    batch_indices = torch.split(indices, BATCH_SIZE)

    average_loss_names = ["Loss data", "Loss remaining weights"]
    average_loss_data = torch.tensor(0.0).to(device)
    average_loss_remaining_weights = torch.tensor(0.0).to(device)

    optimizer = args_optimizers.optimizer

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

    for batch_idx, batch in enumerate(batch_indices):
        data = train_data[batch]
        target = train_labels[batch]
        accumulated_loss = torch.tensor(0.0).to(device)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = MODEL(data)

        loss = criterion(output, target)
        loss_remaining_weights = MODEL.get_remaining_parameters_loss()
        loss_remaining_weights *= pruning_scheduler.get_multiplier()

        # if(epoch > STOP_EPOCH):
        #     loss_remaining_weights *= 0

        average_loss_remaining_weights += loss_remaining_weights.item()
        average_loss_data += loss.item()

        accumulated_loss += loss + loss_remaining_weights

        accumulated_loss.backward()
        optimizer.step()

        if (batch_idx + 1) % BATCH_PRINT_RATE == 0 or (batch_idx + 1) == total_data_len:
            update_args_display_model_statistics(args_display, [average_loss_data, average_loss_remaining_weights], batch_idx, epoch_global)
            display_model_statistics(args_display)
            average_loss_data = torch.tensor(0.0).to(device)
            average_loss_remaining_weights = torch.tensor(0.0).to(device)


def test(args_test: TestData):
    global MODEL, epoch_global, BATCH_SIZE
    MODEL.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    device = get_device()

    test_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device, dtype=torch.long)

    test_data = args_test.test_data
    test_labels = args_test.test_labels
    total_data_len = len(test_data)

    with torch.no_grad():
        batch_indices = torch.split(
            torch.arange(total_data_len, device=get_device()), BATCH_SIZE
        )
        for batch in batch_indices:
            data = test_data[batch]
            target = test_labels[batch]
            output = MODEL(data)
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum()

    test_loss /= total_data_len
    accuracy = 100.0 * correct / total_data_len
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{total_data_len} ({accuracy:.0f}%)\n"
    )

    wandb.log({
        'epoch': epoch_global,
        'test_loss': test_loss,
        'accuracy': accuracy,
    })



MODEL: ModelCifar10Conv2
pruning_scheduler: PruningScheduler
epoch_global = 0
BATCH_SIZE = 128

def run_cifar10_conv2():
    global MODEL, epoch_global, BATCH_SIZE, pruning_scheduler
    configs_layers_initialization_all_kaiming_sqrt5()

    stop_epoch = 20
    num_epochs = 30

    train_data, train_labels, test_data, test_labels = preprocess_cifar10_data_tensors_on_GPU()
    MODEL = ModelCifar10Conv2(ConfigsNetworkMasks(mask_pruning_enabled=True, mask_flipping_enabled=False, weights_training_enabled=True)).to(get_device())
    pruning_scheduler = PruningScheduler(exponent_constant=2, pruning_target=0.0025, epochs_target=stop_epoch, total_parameters=MODEL.get_parameters_total_count())
    weight_bias_params, pruning_params, flipping_params = get_model_parameters_and_masks(MODEL)

    lr_weight_bias = 0.0008
    lr_pruning_params = 0.001
    lr_flipping_params = 0.001

    scheduler_decay_while_pruning = 0.8
    scheduler_decay_after_pruning = 0.8

    wandb.init(project='Dump', config={
        'total_epochs': num_epochs,
        'lr_weight_bias': lr_weight_bias,
        'lr_pruning_params': lr_pruning_params,
        'lr_flipping_params': lr_flipping_params,
        'scheduler_decay_while_pruning': scheduler_decay_while_pruning,
        'scheduler_decay_after_pruning': scheduler_decay_after_pruning,
        'scaled_training_loss': True
    })

    optimizer = torch.optim.AdamW([
                        {'params': weight_bias_params, 'lr': lr_weight_bias},
                        {'params': pruning_params, 'lr': lr_pruning_params},
                        {'params': flipping_params, 'lr': lr_flipping_params}
    ])

    def lambda_lr_weight_bias(epoch):
        if epoch < stop_epoch:
            return scheduler_decay_while_pruning ** (epoch)
        else:
            return scheduler_decay_after_pruning ** (epoch-stop_epoch)


    def lambda_lr_pruning(epoch):
        if epoch < stop_epoch:
            return 1
        else:
            return scheduler_decay_after_pruning ** (epoch - stop_epoch)


    lambda_lr_weight_bias = lambda_lr_weight_bias
    lambda_lr_pruning_params = lambda_lr_pruning
    lambda_lr_flipping_params = lambda_lr_pruning
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda_lr_weight_bias, lambda_lr_pruning_params, lambda_lr_flipping_params])

    for epoch in range(1, num_epochs + 1):
        epoch_global = epoch
        train(ArgsTrain(train_data, train_labels), ArgsOptimizers(optimizer))
        test(TestData(test_data, test_labels))
        scheduler.step()

    print("Training complete")