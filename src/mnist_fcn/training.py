import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.others import get_device
from .model_fcn import ModelMnistFNN
import numpy as np

from ..constants import WEIGHTS_ATTR, BIAS_ATTR, WEIGHTS_PRUNING_ATTR, WEIGHTS_FLIPPING_ATTR

exp = 0

def balancer_parameters(network_loss: float, regularization_loss: float, scale:float = 1, ratio: float = 1) -> tuple[float, float]:
    """
    Balances the network losses in the desired ratio
    :param network_loss: ...
    :param regularization_loss: ...
    :param scale: represents the value of the final network loss
    :param ratio: represents the ratio: regularization_loss / network_loss
    :return:
    """
    a = scale / network_loss
    b = (a * ratio * network_loss) / regularization_loss
    return a, b

def train(model: ModelMnistFNN, train_loader, optimizer, epoch):
    global exp
    model.train()
    criterion = nn.CrossEntropyLoss()
    device = get_device()

    avg_loss_masks = 0
    avg_loss_images = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        accumulated_loss = torch.tensor(0.0).to(device)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss_masks = model.get_remaining_parameters_loss()
        loss_masks = loss_masks * 1

        accumulated_loss += loss
        accumulated_loss += loss_masks

        avg_loss_masks += loss_masks.item()
        avg_loss_images += loss.item()

        accumulated_loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}]')
            total, remaining = model.get_parameters_pruning_statistics()
            pruned_percent = remaining / total
            print(f'Masked weights percentage: {pruned_percent*100:.2f}%,Loss pruned: {loss_masks.item()}, Loss data: {loss.item()}')
            total, remaining = model.get_parameters_flipped_statistics()
            flip_percentage = remaining / total
            print(f'Flipped weights percentage: {flip_percentage*100:.2f}%')

    avg_loss_masks /= len(train_loader.dataset)
    avg_loss_images /= len(train_loader.dataset)

    # if(avg_loss_masks > avg_loss_images):
    #     exp += 1
    #     print("EXPONENT INCREASED")

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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    batch_size = 128
    lr_weight_bias = 0.005
    lr_custom_params = 0.01
    num_epochs = 100

    # Download and load the training data
    train_dataset = datasets.MNIST(root='./data', train=True,
                                   download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Download and load the test data
    test_dataset = datasets.MNIST(root='./data', train=False,
                                  download=True, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    configs_network_masks = {
        'mask_pruning_enabled': True,
        'weights_training_enabled': True,
        'mask_flipping_enabled': True,
    }
    # Instantiate the network, optimizer, etc.

    weight_bias_params = []
    prune_params = []
    flip_params = []

    model = ModelMnistFNN(configs_network_masks).to(get_device())

    for name, param in model.named_parameters():
        if WEIGHTS_ATTR in name or BIAS_ATTR in name:
            weight_bias_params.append(param)
        if WEIGHTS_PRUNING_ATTR in name:
            prune_params.append(param)
        if WEIGHTS_FLIPPING_ATTR in name:
            flip_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': weight_bias_params, 'lr': lr_weight_bias},
        {'params': prune_params, 'lr': lr_custom_params},
        {'params': flip_params, 'lr': lr_custom_params * 5}
    ])

    lambda_lr_weight_bias = lambda epoch: 0.8 ** (epoch // 2)
    lambda_lr_prune_params = lambda epoch: 1.1 ** (epoch // 2)
    lambda_lr_flip_params = lambda epoch: 1 ** (epoch // 2)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda_lr_weight_bias, lambda_lr_prune_params, lambda_lr_flip_params])

    for epoch in range(1, num_epochs + 1):
        # Toggle mask as needed
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)
        scheduler.step()

    print("Training complete")
  
