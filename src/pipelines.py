from src.cifar10_resnet18.train_model import run_cifar10_resnet18
from src.cifar10_resnet50.train_resnet50 import run_cifar10_resnet50
from src.mnist_lenet300.training import run_mnist


#For FC

def pipeline_mnist():
    run_mnist()

def pipeline_cifar10_resnet18():
    run_cifar10_resnet18()

def pipeline_cifar10_resnet50():
    run_cifar10_resnet50()

