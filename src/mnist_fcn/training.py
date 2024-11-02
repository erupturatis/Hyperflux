import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.others import get_device
from .model_fcn import ModelMnistFNN
import numpy as np

from ..constants import WEIGHTS_ATTR, BIAS_ATTR, WEIGHTS_PRUNING_ATTR, WEIGHTS_FLIPPING_ATTR
from ..data_preprocessing import preprocess_mnist
from ..layers import ConfigsNetworkMasks

STOP_EPOCH = 10

def train(model: ModelMnistFNN, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    device = get_device()

    average_loss_masks = 0
    average_loss_dataset = 0
    batch_print_rate = 100

    for batch_idx, (data, target) in enumerate(train_loader):
        accumulated_loss = torch.tensor(0.0).to(device)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss_remaining_weights = model.get_remaining_parameters_loss()
        loss_remaining_weights = loss_remaining_weights * (epoch ** 1.5)

        if(epoch > STOP_EPOCH):
            loss_remaining_weights *= 0

        accumulated_loss += loss
        accumulated_loss += loss_remaining_weights

        average_loss_masks += loss_remaining_weights.item()
        average_loss_dataset += loss.item()

        accumulated_loss.backward()
        optimizer.step()

        if batch_idx % batch_print_rate == 0:
            average_loss_masks /= batch_print_rate
            average_loss_dataset /= batch_print_rate

            print(f'Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}]')
            total, remaining = model.get_parameters_pruning_statistics()
            pruned_percent = remaining / total
            print(f'Masked weights percentage: {pruned_percent*100:.2f}%,Loss pruned: {loss_remaining_weights.item()}, Loss data: {loss.item()}')
            total, remaining = model.get_parameters_flipped_statistics()
            flip_percentage = remaining / total
            print(f'Flipped weights percentage: {flip_percentage*100:.2f}%')

            average_loss_masks = 0
            average_loss_dataset = 0

def test(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(get_device()), target.to(get_device())
            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)      # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    global exp
    if accuracy <= 97.5:
        # exp += 1
        # print("EXPONENT INCREASED")
        pass

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({100. * correct / len(test_loader.dataset):.0f}%)\n')


def run_mnist():
    # Define transformations for the training and testing data
    train_loader, test_loader = preprocess_mnist()

    lr_weight_bias = 0.005
    lr_pruning_params = 0.01
    lr_flipping_params = 0.02
    num_epochs = 100

    weight_bias_params = []
    prune_params = []
    flip_params = []

    model = ModelMnistFNN(ConfigsNetworkMasks(mask_pruning_enabled=True, mask_flipping_enabled=True, weights_training_enabled=True)).to(get_device())
    for name, param in model.named_parameters():
        if WEIGHTS_ATTR in name or BIAS_ATTR in name:
            weight_bias_params.append(param)
        if WEIGHTS_PRUNING_ATTR in name:
            prune_params.append(param)
        if WEIGHTS_FLIPPING_ATTR in name:
            flip_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': weight_bias_params, 'lr': lr_weight_bias},
        {'params': prune_params, 'lr': lr_pruning_params},
        {'params': flip_params, 'lr': lr_flipping_params}
    ])

    def lambda_lr_weight_bias(epoch):
        if epoch < STOP_EPOCH:
            return (0.9 ** epoch)
        else:
            return (0.9 ** (epoch-STOP_EPOCH))

    def lambda_lr_pruning_params(epoch):
        if epoch < STOP_EPOCH:
            return 1
        else:
            return 0.5

    lambda_lr_weight_bias = lambda_lr_weight_bias
    lambda_lr_pruning_params = lambda_lr_pruning_params
    lambda_lr_flipping_params = lambda_lr_pruning_params
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda_lr_weight_bias, lambda_lr_pruning_params, lambda_lr_flipping_params])

    for epoch in range(1, num_epochs + 1):
        # Toggle mask as needed
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)
        scheduler.step()

    print("Training complete")
  
