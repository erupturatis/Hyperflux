import argparse
import sys
from src.mnist_lenet300.train_pruned_lenet300_mnist_adam import train_pruned_lenet300_mnist_adam
from src.mnist_lenet300.train_sparsity_curves_adam import run_lenet300_mnist_adam_sparsity_curve
from src.mnist_lenet300.train_sparsity_curves_fixed_lr import run_lenet300_mnist_adam_sparsity_curve_lrs
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
from src.resnet50_cifar100.train_pruned_resnet50_cifar100_decay_custom import train_resnet50_cifar100_sparse_model_custom_decay
from src.resnet50_cifar100.train_scratch_resnet50_cifar100 import train_resnet50_cifar100_from_scratch
from src.resnet50_cifar100.train_scratch_resnet50_cifar100_multistep import train_resnet50_cifar100_from_scratch_multistep
from src.resnet50_imagenet1k.train_pruned_resnet50_imagenet import train_resnet50_imagenet_sparse_model
from src.vgg19_cifar10.train_pruned_vgg19_cifar10 import train_vgg19_cifar10_sparse_model
from src.vgg19_cifar10.train_scratch_vgg19_cifar10_multistep import train_vgg19_cifar10_from_scratch_multistep
from src.vgg19_cifar100.train_pruned_vgg19_cifar100 import train_vgg19_cifar100_sparse_model
from src.vgg19_cifar100.train_scratch_vgg19_cifar100 import train_vgg19_cifar100_from_scratch
from src.vgg19_cifar100.train_scratch_vgg19_cifar100_multistep import train_vgg19_cifar100_from_scratch_multistep

INITIAL_LR = 0.1
HIGH_LR = 0.01
MID_LR = 0.001
LOW_LR = 0.0001

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for various models and datasets.')
    parser.add_argument('--model', type=str, required=True, choices=['vgg19', 'resnet18', 'resnet50'],
                        help='Model to train.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to use for training. E.g., cifar10, cifar100, imagenet.')
    parser.add_argument('--training_type', type=str, required=True, choices=['pruned', 'scratch'],
                        help='Type of training: pruned or scratch.')
    parser.add_argument('--target_sparsity', type=float, required=True,
                        help='Target sparsity percentage.')
    parser.add_argument('--version', type=str, required=True, choices=['stable_version1', 'stable_version2'],
                        help='Version of the training setup.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to resume training from a checkpoint.')
    parser.add_argument('--notes', type=str, default='',
                        help='Additional notes for the training run.')
    parser.add_argument('--pruning_end', type=int, default=None,
                        help='Pruning end epoch.')
    parser.add_argument('--regrowing_end', type=int, default=None,
                        help='Regrowing end epoch.')
    parser.add_argument('--lr_flow_params_decay_regrowing', type=float, default=None,
                        help='Decay rate for learning rate during regrowing.')
    parser.add_argument('--start_lr_pruning', type=float, default=None,
                        help='Starting learning rate for pruning.')
    parser.add_argument('--end_lr_pruning', type=float, default=None,
                        help='Ending learning rate for pruning.')
    parser.add_argument('--reset_lr_pruning', type=float, default=None,
                        help='Reset learning rate after pruning.')
    parser.add_argument('--end_lr_regrowth', type=float, default=None,
                        help='Ending learning rate for regrowth.')
    parser.add_argument('--reset_lr_flow_params_scaler', type=int, default=None,
                        help='Scaler for resetting learning rate flow parameters.')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay parameter.')
    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error(f"Unknown arguments: {' '.join(unknown)}")
    return args

def get_training_function(model, dataset, training_type):
    if model == 'vgg19':
        if dataset == 'cifar100':
            if training_type == 'pruned':
                return train_vgg19_cifar100_sparse_model
            elif training_type == 'scratch':
                return train_vgg19_cifar100_from_scratch
        elif dataset == 'cifar10':
            if training_type == 'pruned':
                return train_vgg19_cifar10_sparse_model
            elif training_type == 'scratch':
                return train_vgg19_cifar10_from_scratch_multistep
    elif model == 'resnet18':
        if dataset == 'cifar100':
            if training_type == 'pruned':
                return train_resnet18_cifar100_sparse_model
            elif training_type == 'scratch':
                return train_resnet18_cifar100_from_scratch
        elif dataset == 'cifar10':
            if training_type == 'pruned':
                return train_resnet18_cifar10_sparse_model_adam
            elif training_type == 'scratch':
                return train_resnet18_cifar10_from_scratch
    elif model == 'resnet50':
        if dataset == 'cifar100':
            if training_type == 'pruned':
                return train_resnet50_cifar100_sparse_model
            elif training_type == 'scratch':
                return train_resnet50_cifar100_from_scratch
        elif dataset == 'cifar10':
            if training_type == 'pruned':
                return train_resnet50_cifar10_sparse_model
            elif training_type == 'scratch':
                return train_resnet50_cifar10_from_scratch
        elif dataset == 'imagenet':
            if training_type == 'pruned':
                return train_resnet50_imagenet_sparse_model
            elif training_type == 'scratch':
                raise ValueError(f"No scratch training function for {model} on {dataset}.")
    else:
        raise ValueError(f"Unsupported model: {model}")

def set_defaults_based_on_version(args):
    defaults = {}
    if args.version == 'stable_version1':
        if args.model == 'vgg19' and args.dataset == 'cifar100':
            defaults = {
                "pruning_end": 100,
                "regrowing_end": 160,
                "lr_flow_params_decay_regrowing": 0.8,
                "start_lr_pruning": 2 * HIGH_LR,
                "end_lr_pruning": HIGH_LR / 3,
                "reset_lr_pruning": HIGH_LR / 10,
                "end_lr_regrowth": LOW_LR,
                "reset_lr_flow_params_scaler": 5,
                "weight_decay": 5e-4
            }
        elif args.model == 'vgg19' and args.dataset == 'cifar10':
            defaults = {
                "pruning_end": 100,
                "regrowing_end": 160,
                "lr_flow_params_decay_regrowing": 0.8,
                "start_lr_pruning": 2 * HIGH_LR,
                "end_lr_pruning": HIGH_LR / 3,
                "reset_lr_pruning": HIGH_LR / 10,
                "end_lr_regrowth": LOW_LR,
                "reset_lr_flow_params_scaler": 5,
                "weight_decay": 5e-4
            }
        elif args.model == 'resnet50' and args.dataset == 'cifar100':
            defaults = {
                "pruning_end": 100,
                "regrowing_end": 160,
                "lr_flow_params_decay_regrowing": 0.75,
                "start_lr_pruning": HIGH_LR,
                "end_lr_pruning": HIGH_LR / 3,
                "reset_lr_pruning": HIGH_LR / 10,
                "end_lr_regrowth": LOW_LR,
                "reset_lr_flow_params_scaler": 3,
                "weight_decay": 1e-4
            }
        elif args.model == 'resnet50' and args.dataset == 'cifar10':
            defaults = {
                "pruning_end": 100,
                "regrowing_end": 160,
                "lr_flow_params_decay_regrowing": 0.9,
                "start_lr_pruning": 2 * HIGH_LR,
                "end_lr_pruning": HIGH_LR / 3,
                "reset_lr_pruning": HIGH_LR / 10,
                "end_lr_regrowth": LOW_LR,
                "reset_lr_flow_params_scaler": 5,
                "weight_decay": 5e-4
            }
        elif args.model == 'resnet50' and args.dataset == 'imagenet':
            defaults = {
                "pruning_end": 90,
                "regrowing_end": 150,
                "lr_flow_params_decay_regrowing": 0.75,
                "start_lr_pruning": HIGH_LR,
                "end_lr_pruning": HIGH_LR / 3,
                "reset_lr_pruning": HIGH_LR / 10,
                "end_lr_regrowth": LOW_LR,
                "reset_lr_flow_params_scaler": 3,
                "weight_decay": 1e-4
            }
    elif args.version == 'stable_version2':
        if args.model == 'vgg19' and args.dataset == 'cifar100':
            defaults = {
                "pruning_end": 100,
                "regrowing_end": 160,
                "lr_flow_params_decay_regrowing": 0.75,
                "start_lr_pruning": INITIAL_LR,
                "end_lr_pruning": HIGH_LR / 3,
                "reset_lr_pruning": MID_LR,
                "end_lr_regrowth": LOW_LR,
                "reset_lr_flow_params_scaler": 5,
                "weight_decay": 5e-4
            }
        elif args.model == 'vgg19' and args.dataset == 'cifar10':
            defaults = {
                "pruning_end": 100,
                "regrowing_end": 160,
                "lr_flow_params_decay_regrowing": 0.75,
                "start_lr_pruning": INITIAL_LR,
                "end_lr_pruning": HIGH_LR / 3,
                "reset_lr_pruning": MID_LR,
                "end_lr_regrowth": LOW_LR,
                "reset_lr_flow_params_scaler": 5,
                "weight_decay": 5e-4
            }
        elif args.model == 'resnet50' and args.dataset == 'cifar100':
            defaults = {
                "pruning_end": 100,
                "regrowing_end": 160,
                "lr_flow_params_decay_regrowing": 0.75,
                "start_lr_pruning": INITIAL_LR,
                "end_lr_pruning": HIGH_LR / 3,
                "reset_lr_pruning": MID_LR,
                "end_lr_regrowth": LOW_LR,
                "reset_lr_flow_params_scaler": 5,
                "weight_decay": 5e-4
            }
        elif args.model == 'resnet50' and args.dataset == 'cifar10':
            defaults = {
                "pruning_end": 100,
                "regrowing_end": 160,
                "lr_flow_params_decay_regrowing": 0.75,
                "start_lr_pruning": INITIAL_LR,
                "end_lr_pruning": HIGH_LR / 3,
                "reset_lr_pruning": MID_LR,
                "end_lr_regrowth": LOW_LR,
                "reset_lr_flow_params_scaler": 5,
                "weight_decay": 5e-4
            }
        elif args.model == 'resnet50' and args.dataset == 'imagenet':
            defaults = {
                "pruning_end": 90,
                "regrowing_end": 150,
                "lr_flow_params_decay_regrowing": 0.75,
                "start_lr_pruning": HIGH_LR,
                "end_lr_pruning": HIGH_LR / 3,
                "reset_lr_pruning": HIGH_LR / 10,
                "end_lr_regrowth": LOW_LR,
                "reset_lr_flow_params_scaler": 3,
                "weight_decay": 1e-4
            }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args

def validate_args(args):
    hyperparams_pruned = [
        "pruning_end",
        "regrowing_end",
        "lr_flow_params_decay_regrowing",
        "start_lr_pruning",
        "end_lr_pruning",
        "reset_lr_pruning",
        "end_lr_regrowth",
        "reset_lr_flow_params_scaler",
        "weight_decay"
    ]
    hyperparams_scratch = []
    if args.training_type == 'pruned':
        required_hyperparams = hyperparams_pruned
        irrelevant_hyperparams = hyperparams_scratch
    elif args.training_type == 'scratch':
        required_hyperparams = hyperparams_scratch
        irrelevant_hyperparams = hyperparams_pruned
    else:
        raise ValueError(f"Unsupported training type: {args.training_type}")
    missing = []
    for param in required_hyperparams:
        if getattr(args, param) is None:
            missing.append(param)
    if missing:
        raise ValueError(f"Missing required hyperparameters for training type '{args.training_type}': {', '.join(missing)}")
    irrelevant_set = []
    for param in irrelevant_hyperparams:
        if getattr(args, param) is not None:
            irrelevant_set.append(param)
    if irrelevant_set:
        raise ValueError(f"Irrelevant hyperparameters provided for training type '{args.training_type}': {', '.join(irrelevant_set)}")
    return args

def get_multiplier(args):
    remaining = 100 - args.target_sparsity
    if args.version == 'stable_version1':
        if args.model == 'vgg19' and args.dataset == 'cifar100':
            multiplier = 11 / 13
        elif args.model == 'vgg19' and args.dataset == 'cifar10':
            multiplier = 11 / 13
        elif args.model == 'resnet50' and args.dataset == 'cifar100':
            multiplier = 11 / 13
        elif args.model == 'resnet50' and args.dataset == 'cifar10':
            multiplier = 11 / 13
        elif args.model == 'resnet50' and args.dataset == 'imagenet':
            multiplier = 11 / 13
        else:
            multiplier = 11.5 / 13
    elif args.version == 'stable_version2':
        if args.model == 'vgg19' and args.dataset == 'cifar100':
            multiplier = 10 / 13
        else:
            multiplier = 11.5 / 13
    else:
        multiplier = 11.5 / 13
    return remaining * multiplier

def construct_sparsity_configs(args, target_before_regrowth):
    sparsity_configs_aux = {
        "pruning_end": args.pruning_end,
        "regrowing_end": args.regrowing_end,
        "target_sparsity": target_before_regrowth,
        "lr_flow_params_decay_regrowing": args.lr_flow_params_decay_regrowing,
        "start_lr_pruning": args.start_lr_pruning,
        "end_lr_pruning": args.end_lr_pruning,
        "reset_lr_pruning": args.reset_lr_pruning,
        "end_lr_regrowth": args.end_lr_regrowth,
        "reset_lr_flow_params_scaler": args.reset_lr_flow_params_scaler,
        "weight_decay": args.weight_decay,
        "notes": args.notes
    }
    if args.resume and args.training_type == 'pruned':
        sparsity_configs_aux["resume"] = args.resume
    return sparsity_configs_aux

def main():
    args = parse_args()
    valid_datasets = {
        'vgg19': ['cifar10', 'cifar100'],
        'resnet18': ['cifar10', 'cifar100'],
        'resnet50': ['cifar10', 'cifar100', 'imagenet']
    }
    if args.dataset not in valid_datasets.get(args.model, []):
        raise ValueError(f"Dataset '{args.dataset}' is not valid for model '{args.model}'.")
    args = set_defaults_based_on_version(args)
    try:
        args = validate_args(args)
    except ValueError as e:
        print(f"Argument Validation Error: {e}")
        sys.exit(1)
    try:
        training_function = get_training_function(args.model, args.dataset, args.training_type)
    except ValueError as e:
        print(f"Training Function Error: {e}")
        sys.exit(1)
    target_before_regrowth = get_multiplier(args)
    sparsity_configs_aux = construct_sparsity_configs(args, target_before_regrowth)
    try:
        print("IF TRAINING SPARSE NETWORKS, THE TARGET SPARSITY IS CHOSEN AUTOMATICALLY SO THAT AFTER REGROWTH THE DESIRED INPUT SPARSITY IS REACHED")
        training_function(sparsity_configs_aux)
    except Exception as e:
        print(f"Training Function Execution Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
