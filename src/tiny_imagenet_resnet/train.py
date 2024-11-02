import torch
import torch.nn as nn
import torch.optim as optim
from torch.fx.experimental.migrate_gradual_types.constraint import GetItem
from torch.optim import lr_scheduler
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import cv2
import os
from PIL import Image
import numpy as np
from src.others import get_device, get_root_folder
from src.tiny_imagenet_resnet.resnet18_masked import ResNet, BasicBlock
from torch.optim.lr_scheduler import LambdaLR  

def resize_image(image_path, output_size=224):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

class BicubicResizeTransform:
    def __init__(self, output_size=224):
        self.output_size = output_size
    
    def __call__(self, image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (self.output_size, self.output_size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

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

def train(model, train_loader, optimizer_sgd, optimizer_adam, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    device = get_device()

    for batch_idx, (data, target) in enumerate(train_loader):
        accumulated_loss = torch.tensor(0.0).to(device)

        data, target = data.to(device), target.to(device)
        optimizer_sgd.zero_grad()
        optimizer_adam.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss_masks = model.get_pruned_statistics()

        accumulated_loss += loss
        #accumulated_loss += loss_masks 
        
        accumulated_loss.backward()
        optimizer_sgd.step()
        optimizer_adam.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}]')
            percent = model.get_true_masked_percentage_tensor()  # Assuming this function returns a float
            print(f'Masked weights percentage: {percent*100:.2f}%, Loss pruned: {loss_masks.item()}, Loss data: {loss.item()}')

def run_resnet_tiny():
    num_epochs = 40
    batch_size = 128
    data_dir =r'C:\Users\Statia 1\Desktop\AlexoaieAntonio\data\tiny-imagenet-200'

    data_transforms = {
    'train': transforms.Compose([
        BicubicResizeTransform(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        BicubicResizeTransform(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
    
    
    image_datasets = {
        'train': datasets.ImageFolder(root=f'{data_dir}/train', transform=data_transforms['train']),
        'val': datasets.ImageFolder(root=f'{data_dir}/val', transform=data_transforms['val']),
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4)
    }


    model = ResNet(BasicBlock, layers=[2, 2, 2, 2], num_classes=200, mask_enabled=False, freeze_weights=False, signs_enabled=False).to(get_device())
    weight_bias_params = []
    custom_params = []

    for name, param in model.named_parameters():
        if 'mask_param' in name or 'signs_mask_param' in name:
            custom_params.append(param)
        else:
            weight_bias_params.append(param)


    # optimizer_weights = optim.AdamW(weight_bias_params, lr=0.0009)
    # lambda_lr_weights  = lambda epoch: 0.9 ** (epoch // 5)
    # scheduler_weights = LambdaLR(optimizer_weights, lr_lambda=lambda_lr_weights)

    optimizer_weights = optim.SGD(weight_bias_params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_weights, step_size=7, gamma=0.1)
    

    optimizer_custom_params = optim.AdamW(custom_params, lr= 0.001)
    lambda_lr_custom_params = lambda epoch: 1.5 ** (epoch // 3)
    scheduler_custom_params = LambdaLR(optimizer_custom_params, lr_lambda=lambda_lr_custom_params)

    for epoch in range(1, num_epochs + 1):
        train(model,dataloaders['train'], optimizer_weights, optimizer_custom_params, epoch)
        test(model, dataloaders['val'], save=False)
        scheduler_custom_params.step()
        #scheduler_weights.step()
        exp_lr_scheduler.step()