from dataclasses import dataclass
import torchvision.models as models
import torch
import torch.nn as nn
import wandb

from src.cifar10_resnet18.model_resnet18_attributes import RESNET18_CIFAR10_PYTORCH_REGISTERED_LAYERS_ATTRIBUTES
from src.cifar10_resnet18.model_vanilla_resnet18 import ModelVanillaResnet18
from src.config_layers import configs_layers_initialization_all_kaiming_sqrt5
from src.config_other import WANDB_REGISTER
from src.data_preprocessing import preprocess_cifar10_data_tensors_on_GPU, preprocess_cifar10_resnet_data_tensors_on_GPU
from src.others import get_device, ArgsDisplayModelStatistics, display_model_statistics, \
    update_args_display_model_statistics, prefix_path_with_root
from src.cifar10_resnet18.model_base_resnet18 import ModelBaseResnet18, ConfigsModelBaseResnet18
import kornia.augmentation as K

@dataclass
class TestData:
    test_data: torch.Tensor
    test_labels: torch.Tensor

@dataclass
class ArgsOthers:
    epoch: int

def test(args_test: TestData):
    global BATCH_SIZE, MODEL, epoch_global

    MODEL.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    test_data = args_test.test_data
    test_labels = args_test.test_labels

    test_loss = 0
    correct = 0

    total_data_len = len(test_data)
    with torch.no_grad():
        batch_indices = torch.split(torch.arange(total_data_len, device=get_device()), BATCH_SIZE)

        for batch in batch_indices:
            data = test_data[batch]
            target = test_labels[batch]

            output = MODEL(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total_data_len
    accuracy = 100.0 * correct / total_data_len

    total, remaining = MODEL.get_parameters_pruning_statistics()
    remain_percent = remaining / total * 100

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{total_data_len} ({accuracy:.0f}%)\n",
        f"Remaining parameters: {remain_percent:.2f}%"
    )

    # Log test metrics to WandB


    return accuracy  # Return accuracy for custom table


def test_vanilla_pytorch(model, args_test: TestData):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    test_data = args_test.test_data
    test_labels = args_test.test_labels

    test_loss = 0
    correct = 0

    total_data_len = len(test_data)
    with torch.no_grad():
        batch_indices = torch.split(torch.arange(total_data_len, device=get_device()), BATCH_SIZE)

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
    sparsity = get_sparsity(model, RESNET18_CIFAR10_PYTORCH_REGISTERED_LAYERS_ATTRIBUTES)
    print(f"sparsity: {sparsity}")

    return accuracy  # Return accuracy for custom table


BATCH_SIZE = 128
MODEL: ModelVanillaResnet18
AUGMENTATIONS = nn.Sequential(
    K.RandomCrop((32, 32), padding=4),
    K.RandomRotation(degrees=10.0),
    K.RandomHorizontalFlip(p=0.5),
).to(get_device())
epoch_global: int = 0

def get_sparsity(model, registered_layers):
    """
    Calculates the percentage of non-zero weights in the registered layers of the model.

    Parameters:
        model (torch.nn.Module): The PyTorch model.
        registered_layers (list): List of dictionaries containing layer information.

    Returns:
        float: The sparsity percentage (non-zero weights over total weights * 100).
    """
    total_weights = 0
    nonzero_weights = 0

    # Access the model's state dictionary
    state_dict = model.state_dict()

    for layer_info in registered_layers:
        layer_name = layer_info['name']
        layer_type = layer_info['type']

        # Check if the layer name exists in the state_dict
        if layer_name in state_dict:
            weights = state_dict[layer_name]

            # Count total and non-zero weights
            total_weights += weights.numel()
            nonzero_weights += torch.count_nonzero(weights).item()
        else:
            print(f"Warning: {layer_name} not found in the model's state_dict.")

    if total_weights == 0:
        sparsity = 0.0
    else:
        sparsity = (nonzero_weights / total_weights) * 100

    print(f"Total weights: {total_weights}")
    print(f"Non-zero weights: {nonzero_weights}")
    print(f"Sparsity (% of non-zero weights): {sparsity:.2f}%")

    return sparsity

def run_cifar10_resnet18_vanilla_pytorch():
    print("RUNNING THE VANILLA VERSION")
    # Load your state_dict if needed
    filepath = 'saved_models/model_5epochs.pth'
    filepath = prefix_path_with_root(filepath)
    state_dict = torch.load(filepath)

    # Create the standard ResNet-18 model
    model = models.resnet18()

    # Modify the first convolutional layer
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Remove the first max-pooling layer
    model.maxpool = torch.nn.Identity()

    # Modify the fully connected layer to match CIFAR-10 classes
    model.fc = torch.nn.Linear(512, 10)

    # Load the state_dict into the modified model
    model.load_state_dict(state_dict)

    sparsity = get_sparsity(model, RESNET18_CIFAR10_PYTORCH_REGISTERED_LAYERS_ATTRIBUTES)
    print(f"sparsity: {sparsity}")

    model = model.to(get_device())
    _, _, test_data, test_labels = preprocess_cifar10_resnet_data_tensors_on_GPU()

    num_epochs = 10
    for epoch in range(1, num_epochs):
        test_vanilla_pytorch(model, TestData(test_data, test_labels))



def run_cifar10_resnet18_vanilla_testing():
    configs_layers_initialization_all_kaiming_sqrt5()

    global MODEL, BATCH_SIZE, epoch_global

    configs_model_base_resnet18 = ConfigsModelBaseResnet18(num_classes=10)
    MODEL = ModelVanillaResnet18(configs_model_base_resnet18).to(get_device())
    train_data, train_labels, test_data, test_labels = preprocess_cifar10_resnet_data_tensors_on_GPU()
    num_epochs = 10

    # Training loop
    for epoch in range(1, num_epochs + 1):
        epoch_global = epoch
        test(TestData(test_data, test_labels))

    print("Training complete")
