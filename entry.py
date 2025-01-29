from src.mnist_lenet300.train_pruned_lenet300_mnist_adam import train_pruned_lenet300_mnist_adam
from src.mnist_lenet300.train_sparsity_curves_adam import run_lenet300_mnist_adam_sparsity_curve
from src.resnet18_cifar10.run_existing_resnet18_cifar10 import run_resnet18_cifar10_existing_model
from src.infrastructure.constants import PRUNED_MODELS_PATH
from src.resnet18_cifar10.train_pruned_resnet18_cifar10_adam import train_resnet18_cifar10_sparse_model_adam
from src.resnet18_cifar10.train_scratch_resnet18_cifar10 import train_resnet18_cifar10_from_scratch
from src.resnet18_cifar100.train_pruned_resnet18_cifar100 import train_resnet18_cifar100_sparse_model
from src.resnet18_cifar100.train_scratch_resnet18_cifar100 import train_resnet18_cifar100_from_scratch
from src.resnet50_cifar10.train_pruned_resnet50_cifar10 import train_resnet50_cifar10_sparse_model
from src.resnet50_cifar10.train_pruned_resnet50_cifar10_multistep import train_resnet50_cifar10_sparse_model_multistep
from src.resnet50_cifar10.train_scratch_resnet50_cifar10 import train_resnet50_cifar10_from_scratch
from src.resnet50_cifar10.train_scratch_resnet50_cifar10_multistep import train_resnet50_cifar10_from_scratch_multistep
from src.resnet50_cifar100.train_pruned_resnet50_cifar100 import train_resnet50_cifar100_sparse_model
from src.resnet50_cifar100.train_pruned_resnet50_cifar100_decay_custom import \
    train_resnet50_cifar100_sparse_model_custom_decay
from src.resnet50_cifar100.train_scratch_resnet50_cifar100 import train_resnet50_cifar100_from_scratch
from src.resnet50_cifar100.train_scratch_resnet50_cifar100_multistep import \
    train_resnet50_cifar100_from_scratch_multistep
from src.resnet50_imagenet1k.train_pruned_resnet50_imagenet import train_resnet50_imagenet_sparse_model
from src.vgg19_cifar10.train_pruned_vgg19_cifar10 import train_vgg19_cifar10_sparse_model
from src.vgg19_cifar10.train_scratch_vgg19_cifar10_multistep import train_vgg19_cifar10_from_scratch_multistep
from src.vgg19_cifar100.train_pruned_vgg19_cifar100 import train_vgg19_cifar100_sparse_model
from src.vgg19_cifar100.train_scratch_vgg19_cifar100 import train_vgg19_cifar100_from_scratch
from src.vgg19_cifar100.train_scratch_vgg19_cifar100_multistep import train_vgg19_cifar100_from_scratch_multistep

def resnet50_cifar100_stable(target_sparsity):
    remaining = 100 - target_sparsity
    target_before_regrowth = remaining * 10 / 13 # offsets regrowth
    train_resnet50_cifar100_sparse_model(
        sparsity_configs_aux={
            "pruning_end":100,
            "regrowing_end":160,
            "target_sparsity": target_before_regrowth,
            "lr_flow_params_decay_regrowing": 0.8,
            "start_lr_pruning": 2*1e-2,
            "end_lr_pruning": 1e-2/6,
            "reset_lr_pruning": 1e-2/10,
            "end_lr_regrowth": 1e-4,
            "reset_lr_flow_params_scaler": 5,
            "notes": '''
            testing overnight
            '''
        }
    )

def resnet50_cifar10_stable(target_sparsity):
    remaining = 100 - target_sparsity
    target_before_regrowth = remaining * 10 / 13 # offsets regrowth
    train_resnet50_cifar10_sparse_model(
        sparsity_configs_aux={
            "pruning_end":100,
            "regrowing_end":160,
            "target_sparsity": target_before_regrowth,
            "lr_flow_params_decay_regrowing": 0.9,
            "start_lr_pruning": 2*1e-2,
            "end_lr_pruning": 1e-2/6,
            "reset_lr_pruning": 1e-2/10,
            "end_lr_regrowth": 1e-4,
            "reset_lr_flow_params_scaler": 5,
            "weight_decay": 5e-4,
            "notes": '''
            testing overnight
            '''
        }
    )

def resnet50_imagenet_stable(target_sparsity):
    remaining = 100 - target_sparsity
    target_before_regrowth = remaining * 10 / 13 # offsets regrowth
    train_resnet50_imagenet_sparse_model(
        sparsity_configs_aux={
            "pruning_end": 90,
            "regrowing_end": 150,
            "target_sparsity": 2.75,
            "lr_flow_params_decay_regrowing": 0.8,
            "start_lr_pruning": 1e-2,
            "end_lr_pruning": 1e-2/5,
            "reset_lr_pruning": 1e-2/10,
            "end_lr_regrowth": 1e-4,
            "reset_lr_flow_params_scaler": 1,
            "notes": '''
            testing overnight
            '''
        }
    )



def vgg19_cifar100_setup_stable2(target_sparsity):
    remaining = 100 - target_sparsity
    target_before_regrowth = remaining * 10 / 13 # offsets regrowth
    train_vgg19_cifar100_sparse_model(
        sparsity_configs_aux={
            "pruning_end":100,
            "regrowing_end":160,
            "target_sparsity": target_before_regrowth,
            "lr_flow_params_decay_regrowing": 0.8,
            "start_lr_pruning": 1e-1,
            "end_lr_pruning": 1e-2/3,
            "reset_lr_pruning": 1e-2/10,
            "end_lr_regrowth": 1e-4,
            "reset_lr_flow_params_scaler": 5,
            "notes": '''
            testing overnight
            '''
        }
    )

def vgg19_cifar100_setup_stable(target_sparsity):
    remaining = 100 - target_sparsity
    target_before_regrowth = remaining * 10 / 13 # offsets regrowth
    train_vgg19_cifar100_sparse_model(
        sparsity_configs_aux={
            "pruning_end":100,
            "regrowing_end":160,
            "target_sparsity": target_before_regrowth,
            "lr_flow_params_decay_regrowing": 0.8,
            "start_lr_pruning": 2*1e-2,
            "end_lr_pruning": 1e-2/5,
            "reset_lr_pruning": 1e-2/10,
            "end_lr_regrowth": 1e-4,
            "reset_lr_flow_params_scaler": 5,
            "notes": '''
            testing overnight
            '''
        }
    )

def vgg19_cifar10_setup_stable2(target_sparsity):
    remaining = 100 - target_sparsity
    target_before_regrowth = remaining * 10 / 13 # offsets regrowth
    train_vgg19_cifar10_sparse_model(
        sparsity_configs_aux={
            "pruning_end":100,
            "regrowing_end":160,
            "target_sparsity": target_before_regrowth,
            "lr_flow_params_decay_regrowing": 0.8,
            "start_lr_pruning": 1e-1,
            "end_lr_pruning": 1e-2/3,
            "reset_lr_pruning": 1e-2/10,
            "end_lr_regrowth": 1e-4,
            "reset_lr_flow_params_scaler": 5,
            "notes": '''
            testing overnight
            '''
        }
    )



def vgg19_cifar10_setup_stable(target_sparsity):
    remaining = 100 - target_sparsity
    target_before_regrowth = remaining * 10 / 13 # offsets regrowth
    train_vgg19_cifar10_sparse_model(
        sparsity_configs_aux={
            "pruning_end":100,
            "regrowing_end":160,
            "target_sparsity": target_before_regrowth,
            "lr_flow_params_decay_regrowing": 0.8,
            "start_lr_pruning": 2*1e-2,
            "end_lr_pruning": 1e-2/5,
            "reset_lr_pruning": 1e-2/10,
            "end_lr_regrowth": 1e-4,
            "reset_lr_flow_params_scaler": 5,
            "notes": '''
            testing overnight
            '''
        }
    )


def vgg19_cifar100_setup_customizable(target_sparsity):
    remaining = 100 - target_sparsity
    target_before_regrowth = remaining * 10 / 13 # offsets regrowth
    train_vgg19_cifar100_sparse_model(
        sparsity_configs_aux={
            "pruning_end":100,
            "regrowing_end":160,
            "target_sparsity": target_before_regrowth,
            "lr_flow_params_decay_regrowing": 0.8,
            "start_lr_pruning": 1e-1,
            "end_lr_pruning": 1e-2/3,
            "reset_lr_pruning": 1e-2/10,
            "end_lr_regrowth": 1e-4,
            "reset_lr_flow_params_scaler": 10,
            "notes": '''
            testing overnight
            '''
        }
    )

    train_vgg19_cifar100_sparse_model(
        sparsity_configs_aux={
            "pruning_end":100,
            "regrowing_end":160,
            "target_sparsity": target_before_regrowth,
            "lr_flow_params_decay_regrowing": 0.8,
            "start_lr_pruning": 1e-1,
            "end_lr_pruning": 1e-2/3,
            "reset_lr_pruning": 1e-2/10,
            "end_lr_regrowth": 1e-4,
            "reset_lr_flow_params_scaler": 20,
            "notes": '''
            testing overnight
            '''
        }
    )

    train_vgg19_cifar100_sparse_model(
        sparsity_configs_aux={
            "pruning_end":100,
            "regrowing_end":160,
            "target_sparsity": target_before_regrowth,
            "lr_flow_params_decay_regrowing": 0.7,
            "start_lr_pruning": 1e-1,
            "end_lr_pruning": 1e-2/3,
            "reset_lr_pruning": 1e-2/10,
            "end_lr_regrowth": 1e-4,
            "reset_lr_flow_params_scaler": 30,
            "notes": '''
            testing overnight
            '''
        }
    )

    train_vgg19_cifar100_sparse_model(
        sparsity_configs_aux={
            "pruning_end":100,
            "regrowing_end":160,
            "target_sparsity": target_before_regrowth,
            "lr_flow_params_decay_regrowing": 0.7,
            "start_lr_pruning": 1e-1,
            "end_lr_pruning": 1e-2/3,
            "reset_lr_pruning": 1e-2/10,
            "end_lr_regrowth": 1e-4,
            "reset_lr_flow_params_scaler": 20,
            "notes": '''
            testing overnight
            '''
        }
    )

    train_vgg19_cifar100_sparse_model(
        sparsity_configs_aux={
            "pruning_end":100,
            "regrowing_end":160,
            "target_sparsity": target_before_regrowth,
            "lr_flow_params_decay_regrowing": 0.7,
            "start_lr_pruning": 1e-1,
            "end_lr_pruning": 1e-2/3,
            "reset_lr_pruning": 1e-2/10,
            "end_lr_regrowth": 1e-4,
            "reset_lr_flow_params_scaler": 10,
            "notes": '''
            testing overnight
            '''
        }
    )


def vgg19_cifar10_setup_customizable(target_sparsity):
    remaining = 100 - target_sparsity
    target_before_regrowth = remaining * 10 / 13 # offsets regrowth
    train_vgg19_cifar10_sparse_model(
        sparsity_configs_aux={
            "pruning_end":100,
            "regrowing_end":160,
            "target_sparsity": target_before_regrowth,
            "lr_flow_params_decay_regrowing": 0.8,
            "start_lr_pruning": 1e-1,
            "end_lr_pruning": 1e-2/3,
            "reset_lr_pruning": 1e-2/10,
            "end_lr_regrowth": 1e-4,
            "reset_lr_flow_params_scaler": 5,
            "notes": '''
            testing overnight
            '''
        }
    )

    train_vgg19_cifar10_sparse_model(
        sparsity_configs_aux={
            "pruning_end":100,
            "regrowing_end":160,
            "target_sparsity": target_before_regrowth,
            "lr_flow_params_decay_regrowing": 0.8,
            "start_lr_pruning": 1e-1,
            "end_lr_pruning": 1e-2/3,
            "reset_lr_pruning": 1e-2/10,
            "end_lr_regrowth": 1e-4,
            "reset_lr_flow_params_scaler": 10,
            "notes": '''
            testing overnight
            '''
        }
    )

    train_vgg19_cifar10_sparse_model(
        sparsity_configs_aux={
            "pruning_end":100,
            "regrowing_end":160,
            "target_sparsity": target_before_regrowth,
            "lr_flow_params_decay_regrowing": 0.8,
            "start_lr_pruning": 1e-1,
            "end_lr_pruning": 1e-2/3,
            "reset_lr_pruning": 1e-2/10,
            "end_lr_regrowth": 1e-4,
            "reset_lr_flow_params_scaler": 20,
            "notes": '''
            testing overnight
            '''
        }
    )

    train_vgg19_cifar10_sparse_model(
        sparsity_configs_aux={
            "pruning_end":100,
            "regrowing_end":160,
            "target_sparsity": target_before_regrowth,
            "lr_flow_params_decay_regrowing": 0.7,
            "start_lr_pruning": 1e-1,
            "end_lr_pruning": 1e-2/3,
            "reset_lr_pruning": 1e-2/10,
            "end_lr_regrowth": 1e-4,
            "reset_lr_flow_params_scaler": 30,
            "notes": '''
            testing overnight
            '''
        }
    )

    train_vgg19_cifar10_sparse_model(
        sparsity_configs_aux={
            "pruning_end":100,
            "regrowing_end":160,
            "target_sparsity": target_before_regrowth,
            "lr_flow_params_decay_regrowing": 0.7,
            "start_lr_pruning": 1e-1,
            "end_lr_pruning": 1e-2/3,
            "reset_lr_pruning": 1e-2/10,
            "end_lr_regrowth": 1e-4,
            "reset_lr_flow_params_scaler": 20,
            "notes": '''
            testing overnight
            '''
        }
    )

    train_vgg19_cifar10_sparse_model(
        sparsity_configs_aux={
            "pruning_end":100,
            "regrowing_end":160,
            "target_sparsity": target_before_regrowth,
            "lr_flow_params_decay_regrowing": 0.7,
            "start_lr_pruning": 1e-1,
            "end_lr_pruning": 1e-2/3,
            "reset_lr_pruning": 1e-2/10,
            "end_lr_regrowth": 1e-4,
            "reset_lr_flow_params_scaler": 10,
            "notes": '''
            testing overnight
            '''
        }
    )

if __name__ == '__main__':
    vgg19_cifar100_setup_customizable(99)
    pass
