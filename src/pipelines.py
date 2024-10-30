from src.mnist_fcn.training import run_mnist

def pipeline_mnist_sister():
    run_mnist_sister()

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
    
