import torch

from src.cifar10_resnet18.run_existing_resnet18_cifar10 import run_cifar10_resnet18_existing_model
from src.cifar10_resnet18.train_model_scratch import train_cifar10_resnet18_from_scratch
from src.cifar10_resnet18.train_pruned_resnet18_cifar10_adam import train_cifar10_resnet18_sparse_model_adam
from src.infrastructure.constants import PRUNED_MODELS_PATH

if __name__ == '__main__':
    # train_cifar10_resnet18_sparse_model_adam()
    run_cifar10_resnet18_existing_model(folder=PRUNED_MODELS_PATH, model_name="resnet18c2")

    pass
