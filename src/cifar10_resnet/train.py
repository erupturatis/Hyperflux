from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from src.constants import WEIGHTS_ATTR, BIAS_ATTR, WEIGHTS_PRUNING_ATTR, WEIGHTS_FLIPPING_ATTR
from src.data_preprocessing import preprocess_cifar10_data_tensors_on_GPU, preprocess_cifar10_resnet_data_tensors_on_GPU
from src.layers import ConfigsNetworkMasks
from src.others import get_device, ArgsDisplayModelStatistics, iterator_wrapper, update_args_display_model_statistics
from src.cifar10_resnet.model_base_resnet18 import ModelBaseResnet18, ConfigsModelBaseResnet18
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import kornia.augmentation as K
from src.training_common import get_model_parameters_and_masks

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


def train(args_train: ArgsTrain, args_optimizers: ArgsOptimizers):
    global BATCH_SIZE, AUGMENTATIONS, MODEL, EPOCH
    MODEL.train()

    criterion = nn.CrossEntropyLoss()
    device = get_device()

    train_data = args_train.train_data
    train_labels = args_train.train_labels
    total_data_len = len(train_data)

    optimizer_weights = args_optimizers.optimizer_weights
    optimizer_pruning = args_optimizers.optimizer_pruning
    optimizer_flipping = args_optimizers.optimizer_flipping

    average_loss = torch.tensor(0.0).to(device)
    EPOCH_PRINT_RATE = 100

    indices = torch.randperm(total_data_len, device=device)
    batch_indices = torch.split(indices, BATCH_SIZE)

    average_loss_arr = [torch.tensor(0.0).to(device) for _ in range(2)]
    average_loss_names = ["data", "remaining_weights"]

    args_display: ArgsDisplayModelStatistics = ArgsDisplayModelStatistics(
        BATCH_PRINT_RATE=EPOCH_PRINT_RATE,
        DATA_LENGTH=total_data_len,
        batch_size=BATCH_SIZE,
        average_loss_arr=average_loss_arr,
        average_loss_names=average_loss_names,
        model=MODEL
    )

    iterator_with_display = iterator_wrapper(enumerate(batch_indices), args_display)
    for batch_idx, batch in iterator_with_display:
        data = train_data[batch]
        target = train_labels[batch]
        data = AUGMENTATIONS(data)

        optimizer_weights.zero_grad()
        optimizer_pruning.zero_grad()
        optimizer_flipping.zero_grad()

        output = MODEL(data)
        loss_remaining_weights = MODEL.get_remaining_parameters_loss()

        loss = criterion(output, target)
        loss += loss_remaining_weights

        average_loss += loss
        update_args_display_model_statistics(args_display, batch_idx, EPOCH)

        loss.backward()

        optimizer_weights.step()
        optimizer_pruning.step()
        optimizer_flipping.step()


def test(args_test: TestData):
    global BATCH_SIZE, MODEL, EPOCH

    MODEL.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    test_data = args_test.test_data
    test_labels = args_test.test_labels

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

            output = MODEL(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total_data_len
    accuracy = 100.0 * correct / total_data_len
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{total_data_len} ({accuracy:.0f}%)\n"
    )

    # Log test metrics to WandB
    # wandb.log({"epoch": epoch, "test_loss": test_loss, "accuracy": accuracy})

    return accuracy  # Return accuracy for custom table


BATCH_SIZE = 128
MODEL: ModelBaseResnet18
AUGMENTATIONS = nn.Sequential(
    K.RandomCrop((32, 32), padding=4),
    K.RandomRotation(degrees=10.0),
    K.RandomHorizontalFlip(p=0.5),
).to(get_device())
EPOCH: int = 0

def run_cifar10_resnet():
    global MODEL, BATCH_SIZE, EPOCH
    lr_weight_bias = 0.1 # Adjust learning rate as needed
    lr_custom_params = 0.01
    num_epochs = 200
    momentum = 0.9
    weight_decay = 1e-4

    train_data, train_labels, test_data, test_labels = preprocess_cifar10_resnet_data_tensors_on_GPU()
    configs_network_masks = ConfigsNetworkMasks(
        mask_pruning_enabled=False,
        mask_flipping_enabled=False,
        weights_training_enabled=True,
    )
    configs_model_base_resnet18 = ConfigsModelBaseResnet18(num_classes=10)
    MODEL = ModelBaseResnet18(configs_model_base_resnet18, configs_network_masks).to(get_device())

    # wandb.init(
    #     project="Dump",
    #     config={
    #         "batch_size": BATCH_SIZE,
    #         "num_epochs": num_epochs,
    #         "lr_weight_bias": lr_weight_bias,
    #         "lr_custom_params": lr_custom_params,
    #     },
    # )
    # wandb.define_metric("epoch")
    # wandb.define_metric("*", step_metric="epoch")

    # Initialize custom ResNet model
    weight_bias_params, pruning_params, flipping_params = get_model_parameters_and_masks(MODEL)

    # Define optimizers and learning rate schedulers
    optimizer_weights = torch.optim.SGD(
        weight_bias_params,
        lr=lr_weight_bias,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler_weights = CosineAnnealingLR(optimizer_weights, T_max=num_epochs)

    optimizer_pruning = torch.optim.AdamW(pruning_params, lr=lr_custom_params)
    scheduler_pruning = LambdaLR(optimizer_pruning, lr_lambda=lambda epoch: 1 ** epoch)
    optimizer_flipping = torch.optim.AdamW(flipping_params, lr=lr_custom_params)
    scheduler_flipping = LambdaLR(optimizer_flipping, lr_lambda=lambda epoch: 1 ** epoch)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        EPOCH = epoch
        train(ArgsTrain(train_data, train_labels), ArgsOptimizers(optimizer_weights, optimizer_pruning, optimizer_flipping))
        test(TestData(test_data, test_labels))

        scheduler_weights.step()
        scheduler_pruning.step()
        scheduler_flipping.step()

    print("Training complete")
    wandb.finish()
