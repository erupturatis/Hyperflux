import torch
import json
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.others import get_device
from .model_fcn import ModelMnistFNN, ModelMnistFNNAllToAll, ModelMnistFNNAllToAllOther
import wandb

from ..config_layers import configs_layers_initialization_all_kaiming_sqrt0, configs_layers_initialization_all_kaiming_sqrt5
from ..constants import WEIGHTS_ATTR, BIAS_ATTR, WEIGHTS_PRUNING_ATTR, WEIGHTS_FLIPPING_ATTR
from ..data_preprocessing import preprocess_mnist_data_loaders, preprocess_mnist_data_tensors_on_GPU
from ..layers import ConfigsNetworkMasksImportance
from ..mask_functions import INFERENCE, datablob
from ..training_common import get_model_parameters_and_masks

def train(model: ModelMnistFNN, train_data: torch.Tensor, train_labels: torch.Tensor, optimizer, epoch, batch_size=128):
    global SCALER_NETWORK_LOSS
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
        loss_remaining_weights *= SCALER_NETWORK_LOSS

        accumulated_loss += loss
        accumulated_loss += loss_remaining_weights

        average_loss_masks += loss_remaining_weights.item()
        average_loss_dataset += loss.item()

        accumulated_loss.backward()
        optimizer.step()

        if (batch_idx + 1) % BATCH_PRINT_RATE == 0 or (batch_idx + 1) == len(batch_indices) or batch_idx == 0:
            average_loss_masks /= BATCH_PRINT_RATE
            average_loss_dataset /= BATCH_PRINT_RATE

            print(f'Train Epoch: {epoch} [{(batch_idx+1) * batch_size}/{total_data_len}]')

            total, remaining = model.get_parameters_pruning_statistics()
            pruned_percent = remaining / total
            print(f'Masked weights percentage: {pruned_percent*100:.2f}%,Loss pruned: {average_loss_masks}, Loss data: {average_loss_dataset}')
            total, remaining = model.get_parameters_flipped_statistics()
            non_flip_percentage = remaining / total
            print(f'Flipped weights percentage: {non_flip_percentage*100:.2f}%')

            average_loss_masks = 0
            average_loss_dataset = 0

def test(model: ModelMnistFNN, test_data: torch.Tensor, test_labels: torch.Tensor, epoch: int, batch_size=128):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_loss = 0
    correct = 0
    total_data_len = len(test_data)
    INFERENCE["inference"] = True

    with torch.no_grad():
        batch_indices = torch.split(
            torch.arange(total_data_len, device=get_device()), batch_size
        )

        for batch in batch_indices:
            data = test_data[batch]
            target = test_labels[batch]
            output = model(data, True)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)      # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total_data_len
    accuracy = 100.0 * correct / total_data_len
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{total_data_len} ({accuracy:.0f}%)\n"
    )
    INFERENCE["inference"] = False
    _, remaining = model.get_parameters_pruning_statistics()
    print("REMAINING PARAMS", remaining)



SCALER_NETWORK_LOSS = 2

def run_mnist_pruning_curve_experiment():
    global SCALER_NETWORK_LOSS
    for i in range(-2,3):
        SCALER_NETWORK_LOSS = 2 ** i
        run_mnist_with_scaler_loss()

def run_mnist_with_scaler_loss():
    # Define transformations for the training and testing data
    configs_layers_initialization_all_kaiming_sqrt5()
    model = ModelMnistFNN(ConfigsNetworkMasksImportance(mask_pruning_enabled=True, mask_flipping_enabled=False, weights_training_enabled=True)).to(get_device())
    weight_bias_params, prune_params, flip_params = get_model_parameters_and_masks(model)

    lr_weight_bias = 0.005
    lr_pruning_params = 0.001

    num_epochs = 1000

    optimizer = torch.optim.AdamW([
        {'params': weight_bias_params, 'lr': lr_weight_bias, 'weight_decay': 0},
        {'params': prune_params, 'lr': lr_pruning_params, 'weight_decay': 0},
    ])

    lambda_lr_weight_bias = lambda epoch: 0.99 ** epoch
    lambda_lr_pruning = lambda epoch: 1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda_lr_weight_bias, lambda_lr_pruning])
    _, remaining = model.get_parameters_pruning_statistics()
    arr = [remaining.item()]

    train_data, train_labels, test_data, test_labels = preprocess_mnist_data_tensors_on_GPU()
    for epoch in range(1, num_epochs + 1):
        # Toggle mask as needed
        train(model, train_data, train_labels, optimizer, epoch)
        test(model, test_data, test_labels, epoch)
        scheduler.step()
        _, remaining = model.get_parameters_pruning_statistics()
        arr.append(remaining.item())
        with open(f'results_pruning_curve{SCALER_NETWORK_LOSS}.json', 'w') as file:
            json.dump(arr, file)

# 774 params -> 96.67 (0.20%) 30 epochs, stop epoch 15
# 773 params -> 96.70 (0.29%) 30 epochs, stop epoch 15
# 550 params -> 96.00%
