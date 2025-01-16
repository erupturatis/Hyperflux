from typing import Tuple

import torch
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


@dataclass
class Config:
    state_dict_path: str = "/mnt/QNAP/eubar/XAI_paper_antonio/resnet50_imagenet_good_try_34.pth"
    cache_dir: str = "/mnt/QNAP/eubar/data"
    batch_size: int = 1536  # 1024 + 512
    num_workers: int = 50
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_device() -> torch.device:
    """
    Determines the available device (CUDA or CPU).

    Returns:
        torch.device: The device to be used for computations.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_nonzero_and_total_weights(state_dict_path: str) -> Tuple[int, int]:
    """
    Loads the state_dict from the given path and counts the number of non-zero weights and total weights.

    Args:
        state_dict_path (str): Path to the saved state_dict file.

    Returns:
        Tuple[int, int]: Total number of weights and number of non-zero weights.
    """
    try:
        state_dict = torch.load(state_dict_path, map_location='cpu')
    except FileNotFoundError:
        raise FileNotFoundError(f"State dict file not found at {state_dict_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading state dict: {e}")

    total_weights = 0
    nonzero_weights = 0
    skipped_weights = []

    for name, param in state_dict.items():
        # Include all 'weight' parameters excluding those in BatchNorm layers
        if 'weight' in name and 'bn' not in name and 'batchnorm' not in name:
            param_flat = param.view(-1)
            total_weights += param_flat.numel()
            nonzero_weights += torch.count_nonzero(param_flat).item()
        elif 'weight' in name:
            # Parameter has 'weight' but is in a BatchNorm layer or other excluded layer
            skipped_weights.append(name)

    # Diagnostic Output: List skipped weight parameters
    if skipped_weights:
        print("\nSkipped weight parameters (typically BatchNorm layers):")
        for name in skipped_weights:
            print(f" - {name}")

    sparsity = 100 * (1 - nonzero_weights / total_weights) if total_weights > 0 else 0
    print(f"\nTotal weights: {total_weights:,}")
    print(f"Non-zero weights: {nonzero_weights:,}")
    print(f"Sparsity: {sparsity:.2f}%")

    return total_weights, nonzero_weights

def load_pruned_resnet50(state_dict_path: str, device: torch.device) -> torch.nn.Module:
    """
    Loads the pruned state_dict and loads it into a standard ResNet-50 model.

    Args:
        state_dict_path (str): Path to the pruned state_dict file.
        device (torch.device): Device to load the model onto.

    Returns:
        torch.nn.Module: The ResNet-50 model with pruned weights loaded.
    """
    try:
        pruned_state_dict = torch.load(state_dict_path, map_location=device)
    except FileNotFoundError:
        raise FileNotFoundError(f"Pruned state dict file not found at {state_dict_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading pruned state dict: {e}")

    model = models.resnet50()
    model.load_state_dict(pruned_state_dict)
    model.to(device)
    model.eval()  # Set to evaluation mode

    return model


def evaluate_model(model: torch.nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """
    Evaluates the model on the validation dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run the evaluation on.

    Returns:
        float: Validation accuracy in percentage.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total if total > 0 else 0
    print(f'Validation Accuracy: {accuracy:.2f}%')
    return accuracy


class ImageNetDataset(Dataset):
    """
    Custom Dataset for ImageNet data.
    """

    def __init__(self, data, transform=None):
        """
        Initializes the dataset with data and transformations.

        Args:
            data (Dataset): The dataset containing images and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves the image and label at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: Transformed image tensor and its label.
        """
        sample = self.data[idx]
        image = sample['image'].convert("RGB")
        label = sample['label']

        if self.transform:
            image = self.transform(image)

        return image, label


def main(config: Config):
    """
    Main function to execute the evaluation pipeline.

    Args:
        config (Config): Configuration parameters.
    """
    print("Counting non-zero and total weights...")
    count_nonzero_and_total_weights(config.state_dict_path)

    print("Loading pruned ResNet-50 model...")
    model = load_pruned_resnet50(config.state_dict_path, config.device)

    # Define the transformations for validation set
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load the ImageNet dataset using Hugging Face datasets
    print("Loading ImageNet dataset...")
    try:
        dataset = load_dataset("ILSVRC/imagenet-1k", split='validation', cache_dir=config.cache_dir)
    except Exception as e:
        raise RuntimeError(f"Error loading ImageNet dataset: {e}")

    # Create dataset and dataloader for validation
    val_dataset = ImageNetDataset(dataset, transform=val_transforms)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # Evaluate the model on the validation set
    print("Evaluating model...")
    accuracy = evaluate_model(model, val_loader, config.device)
    print(f"Final Validation Accuracy: {accuracy:.2f}%")


def trry():
    config = Config()
    main(config)
