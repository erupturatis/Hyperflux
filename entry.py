from src.utils import get_device, get_project_root
import torch
from src.pipelines import pipeline_mnist, pipeline_test, pipeline_conv2, pipeline_mnist_merged, pipeline_conv4, pipeline_conv6

if __name__ == '__main__':
    # pipeline_cifar10()
    # pipeline_mnist()
    # pipeline_conv2()
    # pipeline_conv4()
    pipeline_conv6()
    pass