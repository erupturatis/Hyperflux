import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import cv2
import os
from PIL import Image
import numpy as np
from src.utils import get_device

# Resize function for bicubic interpolation
def resize_image(image_path, output_size=224):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Custom transform to apply bicubic resizing
class BicubicResizeTransform:
    def __init__(self, output_size=224):
        self.output_size = output_size
    
    def __call__(self, image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (self.output_size, self.output_size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Define data transforms
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

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, device = get_device()):
    import time
    import copy
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase] * 100
            epoch_acc = running_corrects.double() / dataset_sizes[phase] * 100

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    device = get_device()
    data_dir =r'C:\Users\Statia 1\Desktop\AlexoaieAntonio\data\tiny-imagenet-200'  

    image_datasets = {
        'train': datasets.ImageFolder(root=f'{data_dir}/train', transform=data_transforms['train']),
        'val': datasets.ImageFolder(root=f'{data_dir}/val', transform=data_transforms['val']),
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=64, shuffle=False, num_workers=4)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes


    model_ft = models.resnet18(pretrained=True)
    model_ft.avgpool = nn.AdaptiveAvgPool2d(1) 
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 200)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=25)

    # Save the trained model
    torch.save(model_ft.state_dict(), 'resnet18_tiny_imagenet.pth')
