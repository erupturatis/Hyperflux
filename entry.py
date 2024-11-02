from src.others import get_device, get_root_folder
import torch
from src.pipelines import pipeline_mnist, pipeline_cifar10_resnet, pipeline_cifar10_conv2

if __name__ == '__main__':
    # pipeline_cifar10()
    # pipeline_mnist_merged()
    # pipeline_mnist_sister()

    # pipeline_cifar10_resnet()
    # pipeline_cifar10_conv2()
    pipeline_mnist()
    pass