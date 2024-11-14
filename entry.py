from src.pipelines import pipeline_mnist, pipeline_cifar10_resnet18, pipeline_cifar10_conv2, pipeline_cifar10_resnet50
from src.others import get_device, get_root_folder
import torch

if __name__ == '__main__':
    # pipeline_mnist()

    pipeline_cifar10_resnet18()
    # pipeline_cifar10_resnet50()

    # pipeline_cifar10_conv2()
    pass