import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.others import get_device
from .model_fcn import ModelMnistFNN
import wandb

from ..config_layers import configs_layers_initialization_all_kaiming_sqrt0, configs_layers_initialization_all_kaiming_sqrt5
from ..constants import WEIGHTS_ATTR, BIAS_ATTR, WEIGHTS_PRUNING_ATTR, WEIGHTS_FLIPPING_ATTR
from ..data_preprocessing import preprocess_mnist_data_loaders, preprocess_mnist_data_tensors_on_GPU
from ..layers import ConfigsNetworkMasksImportance
from ..training_common import get_model_parameters_and_masks

EXPONENT_CONSTANT = 1.8
INTERVALS = [[0,8], [12, 16], [22, 24], [30, 32], [35, 36]]
# INTERVALS = [[0, 11]]
ITER = 0

def train(model: ModelMnistFNN, train_data: torch.Tensor, train_labels: torch.Tensor, optimizer, epoch, batch_size=128):
    global ITER
    model.train()
    criterion = nn.CrossEntropyLoss()
    device = get_device()

    average_loss_masks = 0
    average_loss_dataset = 0
    BATCH_PRINT_RATE = 100

    total_data_len = len(train_data)
    indices = torch.randperm(total_data_len, device=device)
    batch_indices = torch.split(indices, batch_size)

    in_interval = False
    for intr in INTERVALS:
        if epoch <= intr[1] and intr[0] <= epoch:
            in_interval = True

    if in_interval:
        ITER += 1


    for batch_idx, batch in enumerate(batch_indices):
        data = train_data[batch]
        target = train_labels[batch]
        accumulated_loss = torch.tensor(0.0).to(device)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss_remaining_weights = model.get_remaining_parameters_loss()
        # loss_remaining_weights = loss_remaining_weights * (ITER ** EXPONENT_CONSTANT)
        loss_remaining_weights = loss_remaining_weights / 2

        # if epoch in outside all intervals, multiply by 0
        in_interval = False
        for intr in INTERVALS:
            if epoch <= intr[1] and intr[0] <= epoch:
                in_interval = True

        # if not in_interval:
        #     loss_remaining_weights *= 0

        accumulated_loss += loss
        accumulated_loss += loss_remaining_weights

        average_loss_masks += loss_remaining_weights.item()
        average_loss_dataset += loss.item()

        accumulated_loss.backward()
        optimizer.step()

        if (batch_idx + 1) % BATCH_PRINT_RATE == 0 or (batch_idx + 1) == len(batch_indices):
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

    with torch.no_grad():
        batch_indices = torch.split(
            torch.arange(total_data_len, device=get_device()), batch_size
        )

        for batch in batch_indices:
            data = test_data[batch]
            target = test_labels[batch]
            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)      # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total_data_len
    accuracy = 100.0 * correct / total_data_len
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{total_data_len} ({accuracy:.0f}%)\n"
    )


def run_mnist():
    # Define transformations for the training and testing data
    configs_layers_initialization_all_kaiming_sqrt5()
    train_data, train_labels, test_data, test_labels = preprocess_mnist_data_tensors_on_GPU()
    model = ModelMnistFNN(ConfigsNetworkMasksImportance(mask_pruning_enabled=True, mask_flipping_enabled=False, weights_training_enabled=True)).to(get_device())
    weight_bias_params, prune_params, flip_params = get_model_parameters_and_masks(model)

    lr_weight_bias = 0.005
    lr_pruning_params = 0.001
    lr_flipping_params = 0.001
    num_epochs = 100

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

    lambda_lr_weight_bias = lambda epoch: 0.95 ** epoch
    lambda_lr_pruning_params = lambda epoch: 1
    lambda_lr_flipping_params = lambda epoch: 1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda_lr_weight_bias, lambda_lr_pruning_params, lambda_lr_flipping_params])

    for epoch in range(1, num_epochs + 1):
        # Toggle mask as needed
        train(model, train_data, train_labels, optimizer, epoch)
        test(model, test_data, test_labels, epoch)
        scheduler.step()

    print("Training complete")
    wandb.finish()

