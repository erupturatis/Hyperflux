from src.cifar10_conv2.training import run_cifar10_conv2
from src.cifar10_resnet.train import run_cifar10_resnet
from src.mnist_fcn.training import run_mnist
#For FC
def pipeline_mnist():
    run_mnist()

def pipeline_cifar10_resnet():
    run_cifar10_resnet()

def pipeline_cifar10_conv2():
    run_cifar10_conv2()




