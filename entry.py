import torch
from src.cifar10_resnet18.train_model_adam import run_cifar10_resnet18_adam
from src.cifar10_resnet18.train_sparsity_curves_adam import run_cifar10_resnet18_adam_sparsity_curve
from src.cifar10_resnet18.train_sparsity_curves_sgd import run_cifar10_resnet18_sgd_sparsity_curve
from src.mnist_lenet300.train_model_adam import run_lenet300_mnist_adam
from src.mnist_lenet300.train_model_sgd import run_lenet300_mnist_sgd
from src.mnist_lenet300.train_sparsity_curves_adam import run_lenet300_mnist_adam_sparsity_curve
from src.mnist_lenet300.train_sparsity_curves_sgd import run_lenet300_mnist_sgd_sparsity_curve

if __name__ == '__main__':
    # run_cifar10_resnet18()
    # run_lenet300_mnist_adam()
    # run_lenet300_mnist_sgd()

    # run_lenet300_mnist_adam_sparsity_curve(2, -1, 0)
    # run_lenet300_mnist_sgd_sparsity_curve(2, -1, 0)

    run_cifar10_resnet18_sgd_sparsity_curve(2,-6,8)
    run_cifar10_resnet18_adam_sparsity_curve(2,-6,8)

    pass