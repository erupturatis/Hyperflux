import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb  # Import WandB
from src.constants import WEIGHTS_ATTR, BIAS_ATTR, WEIGHTS_PRUNING_ATTR, WEIGHTS_FLIPPING_ATTR
from src.data_preprocessing import preprocess_cifar10_data_tensors_on_GPU
from src.layers import ConfigsNetworkMasks
from src.others import get_device
from src.cifar10_resnet.model_base_resnet18 import ModelBaseResnet18, ConfigsModelBaseResnet18
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import kornia.augmentation as K


def train(model: ModelBaseResnet18, train_data, train_labels, optimizer_sgd, optimizer_adam, epoch, augmentations, batch_size):
    model.train()
    criterion = nn.CrossEntropyLoss()
    device = get_device()
    total_data_len = len(train_data)
    average_loss = 0.0
    batch_count = 0
    EPOCH_PRINT_RATE = 100

    # Create random indices for batching
    indices = torch.randperm(total_data_len, device=device)
    batch_indices = torch.split(indices, batch_size)

    for batch_idx, batch in enumerate(batch_indices):
        data = train_data[batch]
        target = train_labels[batch]

        data = augmentations(data)


        optimizer_sgd.zero_grad()
        optimizer_adam.zero_grad()

        output = model(data)
        loss_remaining_weights = model.get_remaining_parameters_loss()

        loss = criterion(output, target)
        loss += loss_remaining_weights

        average_loss += loss.item()
        batch_count += 1

        loss.backward()
        optimizer_sgd.step()
        optimizer_adam.step()

        if (batch_idx + 1) % EPOCH_PRINT_RATE == 0 or (batch_idx + 1) == len(batch_indices):
            avg_loss = average_loss / batch_count
            print(f"Train Epoch: {epoch} [{(batch_idx+1)*batch_size}/{total_data_len}]")
            # Get pruning statistics if applicable
            total, remaining = model.get_parameters_pruning_statistics()
            percent = remaining / total * 100
            print(f"Masked weights percentage: {percent:.2f}%, Loss data: {avg_loss}")

            # Log training metrics to WandB
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "masked_weights_percentage": percent,
                    "batch_idx": batch_idx,
                }
            )

            average_loss = 0.0
            batch_count = 0


def test(model, test_data, test_labels, epoch, batch_size):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    test_loss = 0
    correct = 0
    device = get_device()
    total_data_len = len(test_data)

    with torch.no_grad():
        batch_indices = torch.split(
            torch.arange(total_data_len, device=device), batch_size
        )
        for batch in batch_indices:
            data = test_data[batch]
            target = test_labels[batch]
            output = model(data)
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
    wandb.log({"epoch": epoch, "test_loss": test_loss, "accuracy": accuracy})

    return accuracy  # Return accuracy for custom table


def run_cifar10_resnet():
    batch_size = 128  # Adjust as needed based on GPU memory
    lr_weight_bias = 0.1 # Adjust learning rate as needed
    lr_custom_params = 0.01
    num_epochs = 200
    momentum = 0.9
    weight_decay = 1e-4

    train_data, train_labels, test_data, test_labels = preprocess_cifar10_data_tensors_on_GPU()

    wandb.init(
        project="Dump",
        config={
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "lr_weight_bias": lr_weight_bias,
            "lr_custom_params": lr_custom_params,
        },
    )
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")


    # Define data augmentations using Kornia
    augmentations = nn.Sequential(
        K.RandomCrop((32, 32), padding=4),
        K.RandomRotation(degrees=10.0),
        K.RandomHorizontalFlip(p=0.5),
    ).to(get_device())

    # Initialize custom ResNet model
    configs_network_masks = ConfigsNetworkMasks(
        mask_pruning_enabled=False,
        mask_flipping_enabled=False,
        weights_training_enabled=True,
    )
    configs_model_base_resnet18 = ConfigsModelBaseResnet18(num_classes=10)
    model = ModelBaseResnet18(configs_model_base_resnet18, configs_network_masks).to(get_device())

    # Separate model parameters for different optimizers
    weight_bias_params = []
    custom_params = []

    for name, param in model.named_parameters():
        if WEIGHTS_ATTR in name or BIAS_ATTR in name:
            weight_bias_params.append(param)
        elif WEIGHTS_PRUNING_ATTR in name or WEIGHTS_FLIPPING_ATTR in name:
            custom_params.append(param)

    # Define optimizers and learning rate schedulers
    optimizer_sgd = torch.optim.SGD(
        weight_bias_params,
        lr=lr_weight_bias,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler_sgd = CosineAnnealingLR(optimizer_sgd, T_max=num_epochs)

    optimizer_adam = torch.optim.AdamW(custom_params, lr=lr_custom_params)
    scheduler_adam = LambdaLR(optimizer_adam, lr_lambda=lambda epoch: 1.3 ** (epoch // 5))

    # Track metrics for custom table
    table_data = []

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train(model, train_data, train_labels, optimizer_sgd, optimizer_adam, epoch, augmentations, batch_size)
        accuracy = test(
            model, test_data, test_labels, epoch, batch_size
        )
        scheduler_sgd.step()
        scheduler_adam.step()

        # Collect data for custom table
        total, remaining = model.get_parameters_pruning_statistics()
        masked_weights_percentage = remaining / total * 100
        table_data.append([epoch, masked_weights_percentage, accuracy])

    # Create and log a custom table in WandB
    table = wandb.Table(
        data=table_data,
        columns=["Epoch", "Masked Weights Percentage (%)", "Accuracy (%)"],
    )
    wandb.log({"Masked Weights vs Accuracy Table": table})

    print("Training complete")
    wandb.finish()
