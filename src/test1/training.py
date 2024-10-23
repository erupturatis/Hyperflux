import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.utils import get_device
from .network import Net
import numpy as np

def train(model, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    device = get_device()
    for batch_idx, (data, target) in enumerate(train_loader):
        accumulated_loss = torch.tensor(0.0).to(device)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss_masks = model.get_masked_percentage_tensor()

        accumulated_loss += loss
        accumulated_loss += loss_masks

        accumulated_loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}]')
            percent = model.get_masked_percentage_tensor()
            print(f'Masked weights percentage: {percent*100:.2f}%')


def test(model, test_loader, save):
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
    if save == True:
        model.save_weights()
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({100. * correct / len(test_loader.dataset):.0f}%)\n')

def run():
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

    # Instantiate the network, optimizer, etc.
    model = Net(mask_enabled=True, freeze_weights=False, signs_enabled=True).to(get_device())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 40
    for epoch in range(1, num_epochs + 1):
        # Toggle mask as needed
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader, save= True)
    
    print("Training complete")
  
