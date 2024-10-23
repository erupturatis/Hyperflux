import numpy as np 
from src.test1.network import NetSimple
from src.utils import get_device
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn


def cnt_zeros():
    weights = torch.load(r"C:\Users\antoc\OneDrive\Desktop\TOT\Research\xai_good\XAI_paper\nn_weights\model_v1_with_mask.pth")

    zero_count = 0
    total_count = 0

    for key, weight in weights.items():
            if 'weight' in key:  # Only consider weights, skip biases
                total_count += weight.numel()  # Total number of elements
                zero_count += (weight == 0).sum().item()  # Count zeros

    print(f"Total weights: {total_count}")
    print(f"Zero weights: {zero_count}")
    print(f"Percentage of weights that are zero: {100 * zero_count / total_count:.2f}%")

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
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({100. * correct / len(test_loader.dataset):.0f}%)\n')

def run():
    cnt_zeros()
    model = NetSimple()
    model.load_weights(r"XAI_paper\nn_weights\model_v1_with_mask.pth", map_location= get_device() )
    # Define transformations for the training and testing data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and load the training data
    train_dataset = datasets.MNIST(root='./data', train=True,
                                   download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Download and load the test data
    test_dataset = datasets.MNIST(root='./data', train=False,
                                  download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    test(model, test_loader)