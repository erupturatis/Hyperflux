import math
import torch
from matplotlib.pyplot import connect

from src.config_global import configs_layers_initialization_all_kaiming_sqrt0, configs_get_layers_initialization, \
    configs_layers_initialization_all_bad, configs_layers_initialization_all_kaiming_sqrt5
from src.constants import WEIGHTS_ATTR, BIAS_ATTR, WEIGHTS_PRUNING_ATTR, WEIGHTS_FLIPPING_ATTR
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.data_preprocessing import preprocess_cifar10_data_loaders, preprocess_cifar10_data_tensors_on_GPU
from src.layers import ConfigsNetworkMasks
from src.others import get_device
from src.cifar10_conv2.model_conv2 import ModelCifar10Conv2
import numpy as np
from torch.optim.lr_scheduler import StepLR
from src.training_common import get_model_parameters_and_masks
import wandb

STOP_EPOCH = 0
EXPONENT_CONSTANT = 2

def get_model_remaining_parameters_percentage(model:ModelCifar10Conv2):
    total, remaining = model.get_parameters_pruning_statistics()
    return remaining / total

def train(model: ModelCifar10Conv2, train_data: torch.Tensor, train_labels: torch.Tensor, optimizer, epoch, batch_size=128):
    model.train()
    criterion = nn.CrossEntropyLoss()
    device = get_device()

    average_loss_masks = 0
    average_loss_dataset = 0
    BATCH_PRINT_RATE = 100

    total_data_len = len(train_data)
    indices = torch.randperm(total_data_len, device=device)
    batch_indices = torch.split(indices, batch_size)

    for batch_idx, batch in enumerate(batch_indices):
        data = train_data[batch]
        target = train_labels[batch]
        accumulated_loss = torch.tensor(0.0).to(device)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss_remaining_weights = model.get_remaining_parameters_loss()
        loss_remaining_weights *= (epoch ** EXPONENT_CONSTANT) * 0

        if(epoch > STOP_EPOCH):
            loss_remaining_weights *= 0

        average_loss_masks += loss_remaining_weights.item()
        average_loss_dataset += loss.item()

        accumulated_loss += loss + loss_remaining_weights

        accumulated_loss.backward()
        optimizer.step()

        if (batch_idx + 1) % BATCH_PRINT_RATE == 0 or (batch_idx + 1) == len(batch_indices):
            average_loss_masks /= BATCH_PRINT_RATE
            average_loss_dataset /= BATCH_PRINT_RATE

            print(f'Train Epoch: {epoch} [{(batch_idx+1)*batch_size}/{total_data_len}]')

            total, remaining = model.get_parameters_pruning_statistics()
            pruned_percent = remaining / total
            print(f'Masked weights percentage: {pruned_percent*100:.2f}%,Loss pruned: {average_loss_masks}, Loss data: {average_loss_dataset}')
            total, remaining = model.get_parameters_flipped_statistics()
            non_flip_percentage = remaining / total
            print(f'Flipped weights percentage: {non_flip_percentage*100:.2f}%')

            wandb.log({
                'epoch': epoch,
                'train_dataset_loss': average_loss_dataset,
                'remaining_weights_loss': average_loss_masks,
                'pruned_percent': pruned_percent,
                'non_flipped_percent': non_flip_percentage,
                'batch_idx': batch_idx
            })

            average_loss_masks = 0
            average_loss_dataset = 0

def test(model: ModelCifar10Conv2, test_data: torch.Tensor, test_labels: torch.Tensor, epoch: int, batch_size=128):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    device = get_device()
    test_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device, dtype=torch.long)
    total_data_len = len(test_data)

    with torch.no_grad():
        batch_indices = torch.split(
            torch.arange(total_data_len, device=get_device()), batch_size
        )
        for batch in batch_indices:
            data = test_data[batch]
            target = test_labels[batch]
            output = model(data)
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
        'epoch': epoch,
        'test_loss': test_loss,
        'accuracy': accuracy,
    })


def run_cifar10_conv2():
    configs_layers_initialization_all_kaiming_sqrt5()

    train_data, train_labels, test_data, test_labels = preprocess_cifar10_data_tensors_on_GPU()
    model = ModelCifar10Conv2(ConfigsNetworkMasks(mask_pruning_enabled=False, mask_flipping_enabled=False, weights_training_enabled=True)).to(get_device())
    weight_bias_params, pruning_params, flipping_params = get_model_parameters_and_masks(model)

    lr_weight_bias = 0.0008
    lr_pruning_params = 0.001
    lr_flipping_params = 0.001

    num_epochs = 30
    scheduler_decay_while_pruning = 0.8
    scheduler_decay_after_pruning = 0.8

    wandb.init(project='Dump', config={
        'total_epochs': num_epochs,
        'stop_pruning_epoch': STOP_EPOCH,
        'lr_weight_bias': lr_weight_bias,
        'lr_pruning_params': lr_pruning_params,
        'lr_flipping_params': lr_flipping_params,
        'exponent_constant': EXPONENT_CONSTANT,
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
        if epoch < STOP_EPOCH:
            return (scheduler_decay_after_pruning ** (epoch))
            # return 0
        else:
            return (scheduler_decay_after_pruning ** (epoch-STOP_EPOCH))
            # return (scheduler_decay_after_pruning ** (epoch))
            # return 0

    def lambda_lr_pruning(epoch):
        if epoch < STOP_EPOCH:
            # return 0
            return (1 ** epoch)
        else:
            return 0
            return (1 ** epoch)

    lambda_lr_weight_bias = lambda_lr_weight_bias
    lambda_lr_pruning_params = lambda_lr_pruning
    lambda_lr_flipping_params = lambda_lr_pruning
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda_lr_weight_bias, lambda_lr_pruning_params, lambda_lr_flipping_params])

    for epoch in range(1, num_epochs + 1):
        train(model, train_data, train_labels, optimizer, epoch)
        test(model, test_data, test_labels, epoch)
        scheduler.step()

    print("Training complete")