from src.utils import get_device, get_project_root
import torch
from src.pipelines import  pipeline_mnist, pipeline_test, pipeline_cifar10

if __name__ == '__main__':
    # pipeline_cifar10()
    pipeline_mnist()
    pass