import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.others import get_device
from .model_fcn import ModelMnistFNNProbabilistic
import wandb
from ..config_layers import configs_layers_initialization_all_kaiming_sqrt0, configs_layers_initialization_all_kaiming_sqrt5
from ..constants import WEIGHTS_ATTR, BIAS_ATTR, WEIGHTS_PRUNING_ATTR, WEIGHTS_FLIPPING_ATTR, WEIGHTS_BASE_ATTR
from ..data_preprocessing import preprocess_mnist_data_loaders, preprocess_mnist_data_tensors_on_GPU
from ..layers import ConfigsNetworkMasksImportance, ConfigsNetworkMasksProbabilitiesPruneSign
from ..training_common import get_model_parameters_and_masks


def train(train_data: torch.Tensor, train_labels: torch.Tensor, optimizer):
    global MODEL, BATCH_SIZE, epoch_global
    MODEL.train()
    criterion = nn.CrossEntropyLoss()
    device = get_device()

    average_loss_masks = 0
    average_loss_dataset = 0
    BATCH_PRINT_RATE = 100

    total_data_len = len(train_data)
    indices = torch.randperm(total_data_len, device=device)
    batch_indices = torch.split(indices, BATCH_SIZE)

    for batch_idx, batch in enumerate(batch_indices):
        data = train_data[batch]
        target = train_labels[batch]
        accumulated_loss = torch.tensor(0.0).to(device)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = MODEL(data)

        loss = criterion(output, target)
        accumulated_loss += loss

        average_loss_masks += 0
        average_loss_dataset += loss.item()

        accumulated_loss.backward()
        optimizer.step()

        if (batch_idx + 1) % BATCH_PRINT_RATE == 0 or (batch_idx + 1) == len(batch_indices):
            average_loss_masks /= BATCH_PRINT_RATE
            average_loss_dataset /= BATCH_PRINT_RATE

            print(f'Train Epoch: {epoch_global} [{(batch_idx+1) * BATCH_SIZE}/{total_data_len}]')

            average_loss_masks = 0
            average_loss_dataset = 0

def test(test_data: torch.Tensor, test_labels: torch.Tensor):
    global MODEL, epoch_global, BATCH_SIZE
    MODEL.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_loss = 0
    correct = 0
    total_data_len = len(test_data)

    with torch.no_grad():
        batch_indices = torch.split(
            torch.arange(total_data_len, device=get_device()), BATCH_SIZE
        )

        for batch in batch_indices:
            data = test_data[batch]
            target = test_labels[batch]
            output = MODEL.forward_inference(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)      # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total_data_len
    accuracy = 100.0 * correct / total_data_len
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{total_data_len} ({accuracy:.0f}%)\n"
    )
    base, flip, pruned = MODEL.get_parameters_statistics()
    total = base + flip + pruned
    percent_base = base / total * 100.0
    percent_pruned = pruned / total * 100.0
    percent_flipped = flip / total * 100.0
    print(f"Total parameters: {total} | Percent base: {percent_base:.2f}% | Percent flipped: {percent_flipped:.2f}% | Percent pruned: {percent_pruned:.2f}%")



MODEL: ModelMnistFNNProbabilistic
epoch_global = 0
BATCH_SIZE = 128

def run_mnist_probabilistic():
    global MODEL, epoch_global, BATCH_SIZE

    configs_layers_initialization_all_kaiming_sqrt5()
    train_data, train_labels, test_data, test_labels = preprocess_mnist_data_tensors_on_GPU()
    MODEL = ModelMnistFNNProbabilistic(ConfigsNetworkMasksProbabilitiesPruneSign(mask_probabilities_enabled=True, weights_training_enabled=True)).to(get_device())

    lr_weight_bias = 0.005
    lr_probabilities = 0.005
    num_epochs = 100

    weight_bias_params = []
    probabilities_params = []

    for name, param in MODEL.named_parameters():
        if WEIGHTS_ATTR in name or BIAS_ATTR in name:
            weight_bias_params.append(param)
        if WEIGHTS_PRUNING_ATTR in name or WEIGHTS_FLIPPING_ATTR in name or WEIGHTS_BASE_ATTR in name:
            probabilities_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': weight_bias_params, 'lr': lr_weight_bias},
        {'params': probabilities_params, 'lr': lr_probabilities},
    ])

    def lambda_lr_weight_bias(epoch):
        return 1
        # return scheduler_decay_while_pruning ** epoch
    def lambda_lr_probabilities(epoch):
        return 0.9 ** epoch

    lambda_lr_weight_bias = lambda_lr_weight_bias
    lambda_lr_probabilities = lambda_lr_probabilities
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda_lr_weight_bias, lambda_lr_probabilities])

    for epoch in range(1, num_epochs + 1):
        epoch_global = epoch
        train(train_data, train_labels, optimizer)
        test(test_data, test_labels)
        scheduler.step()

    print("Training complete")

