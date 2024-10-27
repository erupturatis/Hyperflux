from src.test1.training import run_mnist
from src.test1.validate_nn import validate
from src.test2.training import run_conv2
from src.test3.training import run_mnist_merged_masks


def pipeline_mnist_merged():
    run_mnist_merged_masks()

#For FC
def pipeline_mnist():
    run_mnist()
    
def pipeline_test():
    validate()
    
#For Conv2 
def pipeline_cifar10():
    run_conv2()
    
