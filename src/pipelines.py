from src.cifar10_conv2.train_conv2 import run_cifar10_conv2
from src.cifar10_resnet18.test_resnet18_vanilla import run_cifar10_resnet18_vanilla_testing, \
    run_cifar10_resnet18_vanilla_pytorch
from src.cifar10_resnet18.train_resnet18 import run_cifar10_resnet18
from src.cifar10_resnet50.train_resnet50 import run_cifar10_resnet50
from src.mnist_fcn.training import run_mnist
from src.mnist_fcn.training_net_flow_experiment import run_mnist_training_net_flow_experiment
from src.mnist_fcn_probabilities.training import run_mnist_probabilistic


#For FC
def pipeline_mnist_probabilistic():
    run_mnist_probabilistic()

def pipeline_mnist_averaged_tests():
    run_mnist_training_net_flow_experiment()

def pipeline_mnist():
    run_mnist()

def pipeline_cifar10_resnet18_vanilla_testing():
    run_cifar10_resnet18_vanilla_pytorch()
    # run_cifar10_resnet18_vanilla_testing()

def pipeline_cifar10_resnet18():
    run_cifar10_resnet18()

def pipeline_cifar10_conv2():
    run_cifar10_conv2()

def pipeline_cifar10_resnet50():
    run_cifar10_resnet50()

