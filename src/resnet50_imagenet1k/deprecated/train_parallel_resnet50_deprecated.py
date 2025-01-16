from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from datasets import load_dataset
from src.config_layers import configs_layers_initialization_all_kaiming_sqrt5
from src.config_other import WANDB_REGISTER
from src.constants import WEIGHTS_ATTR, BIAS_ATTR, WEIGHTS_PRUNING_ATTR, WEIGHTS_FLIPPING_ATTR
from torch.utils.data import Dataset, DataLoader
from src.layers import ConfigsNetworkMasks
from src.others import get_device, ArgsDisplayModelStatistics, display_model_statistics, \
    update_args_display_model_statistics, get_model_remaining_parameters_percentage
from src.imagenet_resnet50.model_base_resnet50 import ModelBaseResnet50, ConfigsModelBaseResnet50
import os
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import kornia.augmentation as K
from src.schedulers import PruningScheduler
from src.training_common import get_model_parameters_and_masks
from PIL import Image
import time 

class ImageNetDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image and label from the dataset
        image = self.data[idx]['image']  # 'image' is already a PIL image
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = self.data[idx]['label']

        return image, label

from torchvision.transforms import InterpolationMode

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BILINEAR),  # Explicit interpolation
    transforms.RandomHorizontalFlip(p=0.5),  # Random flip with 50% probability
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

val_transforms = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),  # Resize shorter edge to 256 with bilinear interpolation
    transforms.CenterCrop(224),  # Center crop to 224x224
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

@dataclass
class ArgsTrain:
    train_loader: DataLoader

@dataclass
class TestData:
    test_loader: DataLoader

@dataclass
class ArgsOptimizers:
    optimizer_weights: torch.optim.Optimizer
    optimizer_pruning: torch.optim.Optimizer
    optimizer_flipping: torch.optim.Optimizer

@dataclass
class ArgsOthers:
    epoch: int
    
def apply_transforms(example, split):
    image = Image.open(example['image']).convert("RGB")  
    transformed_image = train_transforms(image) if split == 'train' else val_transforms(image)
    return {"image": transformed_image, "label": example['label']}


from torch.cuda.amp import autocast, GradScaler


def train(args_train: ArgsTrain, args_optimizers: ArgsOptimizers, epoch_global: int, prune_on : bool):
    global BATCH_SIZE, MODEL, pruning_scheduler, MODEL_MODULE
    MODEL.train()

    criterion = nn.CrossEntropyLoss()
    device = get_device()
    
    train_loader = args_train.train_loader
    total_data_len = len(train_loader.dataset)

    optimizer_weights = args_optimizers.optimizer_weights
    optimizer_pruning = args_optimizers.optimizer_pruning
    optimizer_flipping = args_optimizers.optimizer_flipping

    scaler = GradScaler()  # Initialize GradScaler for mixed precision

    EPOCH_PRINT_RATE = 100

    average_loss_names = ["Loss data", "Loss remaining weights"]
    average_loss_data = torch.tensor(0.0).to(device)
    average_loss_remaining_weights = torch.tensor(0.0).to(device)

    args_display: ArgsDisplayModelStatistics = ArgsDisplayModelStatistics(
        BATCH_PRINT_RATE=EPOCH_PRINT_RATE,
        DATA_LENGTH=total_data_len,
        batch_size=BATCH_SIZE,
        average_loss_names=average_loss_names,
        model=MODEL_MODULE
    )

    total, remaining = MODEL_MODULE.get_parameters_pruning_statistics()
    pruning_scheduler.record_state(remaining)
    pruning_scheduler.step()

    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        if data.size(0) == 0:
            print(f"Empty batch at index {batch_idx}, skipping.")
            continue  

        data = data.to('cuda', non_blocking=True)
        target = target.to('cuda', non_blocking=True)

        optimizer_weights.zero_grad()
        if prune_on:
            optimizer_pruning.zero_grad()
        # optimizer_flipping.zero_grad()

        with autocast():
            output = MODEL(data)

            loss_remaining_weights = MODEL_MODULE.get_remaining_parameters_loss() * (pruning_scheduler.get_multiplier())
                           
            loss_data = criterion(output, target)
            if prune_on:
                loss = loss_data  #+ loss_remaining_weights
            else:
                loss = loss_data 

        average_loss_data += loss_data.item()
        average_loss_remaining_weights += loss_remaining_weights.item()

        lr_weights = optimizer_weights.param_groups[0]['lr']
        lr_pruning = optimizer_pruning.param_groups[0]['lr']
        lr_flipping = optimizer_flipping.param_groups[0]['lr']
        
        if batch_idx % EPOCH_PRINT_RATE == 0 or batch_idx == len(train_loader):
            update_args_display_model_statistics(
                args_display, 
                [average_loss_data, average_loss_remaining_weights], 
                batch_idx, 
                epoch_global, 
                [lr_weights, lr_pruning, lr_flipping]
            )
            display_model_statistics(args_display)
            average_loss_data = 0.0
            average_loss_remaining_weights = 0.0
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes, seconds = divmod(elapsed_time, 60)
            print(f"Epoch {epoch_global}, Batch: {batch_idx} completed in {int(minutes)}m {int(seconds)}s.")
            start_time = time.time()

        # Scale loss and call backward
        scaler.scale(loss).backward()

        # Step the optimizers
        scaler.step(optimizer_weights)
        if prune_on:
            scaler.step(optimizer_pruning)
        #scaler.step(optimizer_flipping)
        scaler.update()  # Update the scaler


def test(args_test: TestData, epoch_global: int):
    global BATCH_SIZE, MODEL, MODEL_MODULE

    MODEL.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    test_loader = args_test.test_loader

    test_loss = 0
    correct = 0

    device = get_device()
    with torch.no_grad():
        for data, target in test_loader:
            if data.size(0) == 0:
                print(f"Empty batch at index {batch_idx} during testing.")
                continue
            data, target = data.to(device, non_blocking = True), target.to(device, non_blocking = True)

            output = MODEL(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            

    test_loss /= len(test_loader.dataset)   # .dataset
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )

    # Log test metrics to WandB
    remain_percent = get_model_remaining_parameters_percentage(MODEL_MODULE)
    if WANDB_REGISTER:
        wandb.log({
            "epoch": epoch_global, 
            "test_loss": test_loss, 
            "accuracy": accuracy, 
            "remaining_parameters": remain_percent 
        })

    return accuracy  # Return accuracy for custom table

MODEL_MODULE = None
BATCH_SIZE = 1024 + 512 # Adjust based on your GPU memory
MODEL: ModelBaseResnet50
pruning_scheduler: PruningScheduler
epoch_global: int = 0

def run_imagenet_resnet50_deprecated():
    print(f"Number of GPUs available: {torch.cuda.device_count()}") 
    configs_layers_initialization_all_kaiming_sqrt5()

    global MODEL, BATCH_SIZE, epoch_global, pruning_scheduler, MODEL_MODULE
    
    lr_weight_bias = 0.001
    
    lr_custom_params = 0.001
                 
    stop_epoch = 0
    num_epochs = 100

    
    momentum = 0.9
    weight_decay = 1e-4



    dataset = load_dataset("ILSVRC/imagenet-1k", cache_dir = "/mnt/QNAP/eubar/data")
    train_dataset = ImageNetDataset(dataset['train'], transform = train_transforms)
    val_dataset = ImageNetDataset(dataset['validation'], transform = val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 50, pin_memory= True)
    val_loader = DataLoader(val_dataset, batch_size= BATCH_SIZE, shuffle=False, num_workers = 50, pin_memory= True)
    print(f"Validation Dataset Size: {len(val_loader.dataset)}")


    configs_network_masks = ConfigsNetworkMasks(
        mask_pruning_enabled= True,
        mask_flipping_enabled=False,
        weights_training_enabled=True,
    )
    configs_model_base_resnet50 = ConfigsModelBaseResnet50(num_classes=1000)  # ImageNet has 1000 classes
    
    MODEL = ModelBaseResnet50(configs_model_base_resnet50, configs_network_masks)
    
    
    pretrained_weights = torch.load("/mnt/QNAP/eubar/XAI_paper_antonio/src/imagenet_resnet50/saved_weights/2resnet50_imagenet_30.pth")
    MODEL.load_state_dict(pretrained_weights)
    # MODEL.load_weights(MODEL)
    
    if torch.cuda.device_count() > 1:   
        MODEL = nn.DataParallel(MODEL, device_ids=[0,1,2,3,4,5])

    MODEL = MODEL.to(get_device())
    MODEL_MODULE = MODEL.module
    
    pruning_scheduler = PruningScheduler(
        exponent_constant=2, 
        pruning_target=0.02, 
        epochs_target=stop_epoch, 
        total_parameters=MODEL_MODULE.get_parameters_total_count()
    )

    if WANDB_REGISTER:
        wandb.init(
            project="resnet50_imagenet",
            config={
                "batch_size": BATCH_SIZE,
                "num_epochs": num_epochs,
                "lr_weight_bias": lr_weight_bias,
                "lr_custom_params": lr_custom_params,
            },
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    # Initialize custom ResNet model
    weight_bias_params, pruning_params, flipping_params = get_model_parameters_and_masks(MODEL_MODULE)

    # Define optimizers and learning rate schedulers
    optimizer_weights = torch.optim.SGD(
        weight_bias_params,
        lr=lr_weight_bias,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    # optimizer_weights = torch.optim.AdamW(weight_bias_params, lr=lr_weight_bias)
    # scheduler_weights = LambdaLR(
    #     optimizer_weights, 
    #     lr_lambda = lambda epoch: 0.9 ** (epoch // 2)
    # )


    scheduler_weights = CosineAnnealingLR(optimizer_weights, T_max=num_epochs, eta_min=1e-6)

    optimizer_pruning = torch.optim.AdamW(pruning_params, lr=lr_custom_params)
    scheduler_decay_after_pruning = 0.9 # 0.8
    scheduler_pruning = LambdaLR(
        optimizer_pruning, 
        lr_lambda=lambda epoch: 1 if epoch < stop_epoch else scheduler_decay_after_pruning ** (epoch - stop_epoch)
    )
        
    optimizer_flipping = torch.optim.AdamW(flipping_params, lr=lr_custom_params)
    scheduler_flipping = LambdaLR(
        optimizer_flipping, 
        lr_lambda=lambda epoch: 1 if epoch < stop_epoch else scheduler_decay_after_pruning ** (epoch - stop_epoch)
    )

    args_optimizers = ArgsOptimizers(
        optimizer_weights=optimizer_weights,
        optimizer_pruning=optimizer_pruning,
        optimizer_flipping=optimizer_flipping
    )
    # e5c716885683
    prune_on = True
    
    test(TestData(val_loader), epoch_global)
    for epoch in range(1, num_epochs + 1):  
        if epoch == 1:
            for param in pruning_params:
                if param.requires_grad == False:
                    print("FUCK")
                    param.requires_grad = True
  
        if epoch % 10 == 0 or epoch == 1:
            MODEL_MODULE.save_weights(f'src/imagenet_resnet50/saved_weights/2resnet50_imagenet_default_{epoch}.pth')
            torch.save(MODEL_MODULE.state_dict(), f'src/imagenet_resnet50/saved_weights/2resnet50_imagenet_{epoch}.pth')    
                 

        
        epoch_global = epoch
        print(f"Epoch {epoch}/{num_epochs}")
        
        train(ArgsTrain(train_loader), args_optimizers, epoch_global, prune_on)
        test(TestData(val_loader), epoch_global)
        
        scheduler_weights.step()
        scheduler_pruning.step()
        scheduler_flipping.step()

        
    print("Training complete")
    if WANDB_REGISTER:
        wandb.finish()
