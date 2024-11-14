from src.cifar10_conv2.train_conv2 import run_cifar10_conv2
from src.cifar10_resnet18.test_resnet18_vanilla import run_cifar10_resnet18_vanilla_testing
from src.cifar10_resnet18.train_resnet18 import run_cifar10_resnet18
from src.cifar10_resnet50.train_resnet50 import run_cifar10_resnet50
from src.mnist_fcn.training import run_mnist

#For FC
def pipeline_mnist():
    run_mnist()

def pipeline_cifar10_resnet18_vanilla_testing():
    run_cifar10_resnet18_vanilla_testing()

def pipeline_cifar10_resnet18():
    run_cifar10_resnet18()

def pipeline_cifar10_conv2():
    run_cifar10_conv2()

def pipeline_cifar10_resnet50():
    run_cifar10_resnet50()

