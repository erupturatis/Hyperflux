from src.cifar10_resnet18_intervals.train_resnet18 import run_cifar10_resnet18_intervals
from src.pipelines import pipeline_mnist, pipeline_cifar10_resnet18, pipeline_cifar10_conv2, pipeline_cifar10_resnet50, \
    pipeline_cifar10_resnet18_vanilla_testing, pipeline_mnist_probabilistic
from src.others import get_device, get_root_folder
import torch

if __name__ == '__main__':
    # pipeline_mnist_probabilistic()
    pipeline_mnist()
    # pipeline_cifar10_resnet18()
    # run_cifar10_resnet18_intervals()
    pass