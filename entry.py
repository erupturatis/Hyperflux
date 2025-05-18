from src.infrastructure.constants import INITIAL_LR
from src.infrastructure.others import TrainingConfigsWithResume
from src.mnist_lenet300.train_pruned_lenet300_bottleneck import train_pruned_lenet300_mnist_bottleneck
from src.mnist_lenet300.train_sparsity_curves_adam import run_lenet300_mnist_adam_sparsity_curve
from src.resnet50_cifar10.train_pruned_resnet50_cifar10 import train_resnet50_cifar10_sparse_model
from src.resnet50_cifar100.train_pruned_resnet50_cifar100 import train_resnet50_cifar100_sparse_model
from src.resnet50_imagenet1k.train_pruned_resnet50_imagenet import train_resnet50_imagenet_sparse_model
from src.vgg19_cifar10.train_pruned_vgg19_cifar10 import train_vgg19_cifar10_sparse_model


def lenet300_bottleneck_experiment():
    train_pruned_lenet300_mnist_bottleneck()

def lenet300_convergence_experiments():
    run_lenet300_mnist_adam_sparsity_curve(2, -2,2)

def resnet50_imagenet_95_sparsity():
    defaults: TrainingConfigsWithResume = {
        "pruning_end": 80,
        "regrowing_end": 100,
        "target_sparsity": 96, # 4% remaining parameters
        "resume": "resnet50_imagenet_baseline.pth",
        "lr_flow_params_decay_regrowing": 0.75,
        "start_lr_pruning": 0.01,
        "end_lr_pruning": 0.01 / 3,
        "reset_lr_pruning": 0.01 / 10,
        "end_lr_regrowth": 0.0001,
        "reset_lr_flow_params_scaler": 1,
        "weight_decay": 1e-4,
        "notes": "Running imagenet1k"
    }
    train_resnet50_imagenet_sparse_model(defaults)

def resnet50_cifar100_98_sparsity():
    defaults: TrainingConfigsWithResume = {
        "pruning_end": 100,
        "regrowing_end": 160,
        "lr_flow_params_decay_regrowing": 0.75,
        "start_lr_pruning": INITIAL_LR,
        "end_lr_pruning": INITIAL_LR / 30,
        "reset_lr_pruning": INITIAL_LR / 100,
        "end_lr_regrowth": INITIAL_LR / 1000,
        "reset_lr_flow_params_scaler": 1,
        "weight_decay": 5e-4,
        "target_sparsity": 100 - 2 * 12.5/13,
        "resume": "resnet50_cifar10_accuracy94.64%",
        "notes": "resnet50 cifar10"
    }
    train_resnet50_cifar100_sparse_model(defaults)

def resnet50_cifar10_98_sparsity():
    defaults: TrainingConfigsWithResume = {
        "pruning_end": 100,
        "regrowing_end": 160,
        "lr_flow_params_decay_regrowing": 0.75,
        "start_lr_pruning": INITIAL_LR,
        "end_lr_pruning": INITIAL_LR / 30,
        "reset_lr_pruning": INITIAL_LR / 100,
        "end_lr_regrowth": INITIAL_LR / 1000,
        "reset_lr_flow_params_scaler": 1,
        "weight_decay": 5e-4,
        "target_sparsity": 100 - 2 * 12.5/13,
        "resume": "resnet50_cifar10_accuracy94.64%",
        "notes": "resnet50 cifar10"
    }
    train_resnet50_cifar10_sparse_model(defaults)

def vgg19_cifar10_98_sparsity():
    defaults: TrainingConfigsWithResume = {
        "pruning_end": 100,
        "regrowing_end": 160,
        "lr_flow_params_decay_regrowing": 0.75,
        "start_lr_pruning": INITIAL_LR,
        "end_lr_pruning": INITIAL_LR / 30,
        "reset_lr_pruning": INITIAL_LR / 100,
        "end_lr_regrowth": INITIAL_LR / 1000,
        "reset_lr_flow_params_scaler": 1,
        "weight_decay": 5e-4,

        "target_sparsity": 100 - 2 * 12.5/13,
        "resume": "resnet50_cifar10_accuracy94.64%",
        "notes": "resnet50 cifar10"
    }
    train_vgg19_cifar10_sparse_model(defaults)

def vgg19_cifar100_98_sparsity():
    defaults: TrainingConfigsWithResume = {
        "pruning_end": 100,
        "regrowing_end": 160,
        "lr_flow_params_decay_regrowing": 0.75,
        "start_lr_pruning": INITIAL_LR,
        "end_lr_pruning": INITIAL_LR / 30,
        "reset_lr_pruning": INITIAL_LR / 100,
        "end_lr_regrowth": INITIAL_LR / 1000,
        "reset_lr_flow_params_scaler": 1,
        "weight_decay": 5e-4,

        "target_sparsity": 100 - 2 * 12.5/13,
        "resume": "resnet50_cifar10_accuracy94.64%",
        "notes": "resnet50 cifar10"
    }

def main():
    resnet50_cifar10_98_sparsity()
    resnet50_cifar100_98_sparsity()
    vgg19_cifar10_98_sparsity()
    vgg19_cifar100_98_sparsity()
    pass

if __name__ == "__main__":
    main()