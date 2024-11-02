import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.constants import WEIGHTS_ATTR, BIAS_ATTR, WEIGHTS_PRUNING_ATTR, WEIGHTS_FLIPPING_ATTR
from src.layers import ConfigsNetworkMasks
from src.others import get_device
from src.cifar10_resnet.model_base_resnet18 import ModelBaseResnet18, ConfigsModelBaseResnet18
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, ColorJitter, RandomRotation, RandomErasing
from torch.optim.lr_scheduler import LambdaLR  

def train(model: ModelBaseResnet18, train_loader, optimizer_sgd, optimizer_adam, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    device = get_device()

    average_loss = 0.0

    EPOCH_PRINT_RATE = 100
    for batch_idx, (data, target) in enumerate(train_loader):
        accumulated_loss = torch.tensor(0.0).to(device)

        data, target = data.to(device), target.to(device)
        optimizer_sgd.zero_grad()
        optimizer_adam.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss_masks = model.get_remaining_parameters_loss()

        accumulated_loss += loss
        average_loss += loss.item()

        accumulated_loss += loss_masks 
        
        accumulated_loss.backward()
        optimizer_sgd.step()
        optimizer_adam.step()
        
        if batch_idx % EPOCH_PRINT_RATE == 0:
            average_loss /= EPOCH_PRINT_RATE
            print(f'Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}]')
            total, remaining = model.get_parameters_pruning_statistics()
            percent = remaining / total
            print(f'Masked weights percentage: {percent*100:.2f}%, Loss pruned: {loss_masks.item()}, Loss data: {average_loss}')
            average_loss = 0.0

def test(model, test_loader, save=False):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_loss = 0
    correct = 0
    device = get_device()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)      # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({100. * correct / len(test_loader.dataset):.0f}%)\n')

def run_cifar10_resnet():
    img_size=224
    crop_size = 224
    mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
    batch_size = 128
    lr_weight_bias = 0.001
    lr_custom_params = 0.003
    num_epochs = 40
    momentum = 0.9
    weight_decay = 4e-4

    # Load CIFAR-10 dataset
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform= transforms.Compose(
    [
     transforms.Resize(img_size),
     transforms.RandomRotation(20),
     transforms.RandomHorizontalFlip(0.1),
     transforms.ColorJitter(brightness=0.1,contrast = 0.1 ,saturation =0.1 ),
     transforms.RandomAdjustSharpness(sharpness_factor = 2, p = 0.1),
     transforms.ToTensor(),
     transforms.Normalize(mean,std),
     transforms.RandomErasing(p=0.75,scale=(0.02, 0.1),value=1.0, inplace=False)])
)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose(
[
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)]))
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize custom ResNet model for CIFAR-10 with 10 classes

    configs_network_masks = ConfigsNetworkMasks(mask_pruning_enabled=False, mask_flipping_enabled=False, weights_training_enabled=True)
    configs_model_base_resnet18 = ConfigsModelBaseResnet18(num_classes=10)
    model = ModelBaseResnet18(configs_model_base_resnet18, configs_network_masks).to(get_device())

    weight_bias_params = []
    custom_params = []

    for name, param in model.named_parameters():
        if WEIGHTS_ATTR in name or BIAS_ATTR in name:
            weight_bias_params.append(param)
        if WEIGHTS_PRUNING_ATTR in name or WEIGHTS_FLIPPING_ATTR in name:
            custom_params.append(param)

    optimizer_sgd = torch.optim.AdamW(weight_bias_params, lr = lr_weight_bias)
    optimizer_adam = torch.optim.AdamW(custom_params, lr=lr_custom_params)

    lambda_lr_weight_bias = lambda epoch: 0.9 ** (epoch)
    lambda_lr_custom_params = lambda epoch: 1 ** (epoch // 10)
    scheduler_sgd = LambdaLR(optimizer_sgd, lr_lambda=lambda_lr_weight_bias)
    scheduler_adam = LambdaLR(optimizer_adam, lr_lambda=lambda_lr_custom_params)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, optimizer_sgd, optimizer_adam, epoch)
        test(model, test_loader, save=False)
        scheduler_sgd.step()
        scheduler_adam.step()

    print("Training complete")

