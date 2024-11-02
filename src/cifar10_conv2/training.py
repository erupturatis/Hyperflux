import math
import torch
from matplotlib.pyplot import connect

from src.constants import WEIGHTS_ATTR, BIAS_ATTR, WEIGHTS_PRUNING_ATTR, WEIGHTS_FLIPPING_ATTR
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.data_preprocessing import preprocess_cifar10
from src.layers import ConfigsNetworkMasks
from src.others import get_device
from src.cifar10_conv2.model_conv2 import ModelCifar10Conv2
import numpy as np
from torch.optim.lr_scheduler import StepLR

from src.training_common import get_model_parameters_and_masks

def train(model:ModelCifar10Conv2, train_loader, optimizer, epoch):
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
        loss_masks = model.get_remaining_parameters_loss()
        loss_masks *= (15*epoch)
        if(epoch > 10):
            loss_masks *= 0


        average_loss_masks += loss_masks.item()
        average_loss_dataset += loss.item()

        accumulated_loss += loss + loss_masks

        accumulated_loss.backward()
        optimizer.step()

        if batch_idx % batch_print_rate == 0:
            average_loss_masks /= batch_print_rate
            average_loss_dataset /= batch_print_rate

            print(f'Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}]')
            if batch_idx == 0:
                continue

            total, remaining = model.get_parameters_pruning_statistics()
            percent = remaining / total
            print(f'Masked weights percentage: {percent*100:.2f}%,Loss remaining: {average_loss_masks}, Loss data: {average_loss_dataset}')

            total, remaining = model.get_parameters_flipped_statistics()
            percent = remaining / total
            print(f'Flipped weights percentage: {percent*100:.2f}%,Loss remaining: {average_loss_masks}, Loss data: {average_loss_dataset}')



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

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({100. * correct / len(test_loader.dataset):.0f}%)\n')

def run_cifar10_conv2():

    train_loader, test_loader = preprocess_cifar10()


    model = ModelCifar10Conv2(ConfigsNetworkMasks(mask_pruning_enabled=True, mask_flipping_enabled=True, weights_training_enabled=True)).to(get_device())

    weight_bias_params, pruning_params, flipping_params = get_model_parameters_and_masks(model)
    # custom_params = pruning_params + flipping_params

    lr_weight_bias = 0.0008
    lr_pruning_params = lr_weight_bias * 10
    lr_flipping_params = lr_weight_bias * 20

    # lr_pruning_params = 0
    # lr_flipping_params = 0

    num_epochs = 100
    optimizer = torch.optim.AdamW([
                        {'params': weight_bias_params, 'lr': lr_weight_bias},
                        {'params': pruning_params, 'lr': lr_pruning_params},
                        {'params': flipping_params, 'lr': lr_flipping_params}
    ])

    # lambda_lr_weight_bias = lambda epoch: 0.25 ** (epoch // 2)
    def lambda_lr_weight_bias(epoch):
        if epoch < 10:
            return (0.9 ** epoch)
        else:
            return (0.25 ** ((epoch-10) // 2))

    def lambda_lr_pruning_params(epoch):
        if epoch < 10:
            return 1
        else:
            return 0.5

    lambda_lr_weight_bias = lambda_lr_weight_bias
    lambda_lr_pruning_params = lambda_lr_pruning_params
    lambda_lr_flipping_params = lambda_lr_pruning_params

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda_lr_weight_bias, lambda_lr_pruning_params, lambda_lr_flipping_params])

    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)
        scheduler.step()

    print("Training complete")