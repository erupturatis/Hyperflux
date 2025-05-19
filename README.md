# Hyperflows

## Overview
This script trains various models (VGG19, ResNet18, ResNet50) on datasets like CIFAR-10, CIFAR-100, and ImageNet. It supports both pruned and scratch training configurations.

## Prerequisites
- Python 3.x
- Required dependencies: Install packages listed in `requirements.txt`

## Usage

Basic Command
```bash
python entry.py \
  --model <model_name> \
  --dataset <dataset_name> \
  --training_type <pruned/scratch> \
  --target_sparsity <0-99.9> \
  --version <stable_version1/stable_version2>
```

### Possible Arguments for Training Script

| Argument                          | Type    | Required | Choices                                    | Description |
|------------------------------------|---------|----------|--------------------------------------------|-------------|
| `--model`                         | string  | Yes      | `vgg19`, `resnet18`, `resnet50`            | Model to train. |
| `--dataset`                       | string  | Yes      | `cifar10`, `cifar100`, `imagenet`          | Dataset for training. |
| `--training_type`                  | string  | Yes      | `pruned`, `scratch`                        | Type of training. |
| `--target_sparsity`                | float   | Yes      | N/A                                        | Target sparsity percentage. |
| `--version`                        | string  | Yes      | `stable_version1`, `stable_version2`       | Version of training setup. |
| `--resume`                         | string  | No       | N/A                                        | Path to resume training. |
| `--notes`                          | string  | No       | N/A                                        | Additional notes for training run. |
| `--pruning_end`                    | int     | No       | N/A                                        | Pruning end epoch. |
| `--regrowing_end`                  | int     | No       | N/A                                        | Regrowing end epoch. |
| `--lr_flow_params_decay_regrowing`  | float   | No       | N/A                                        | Decay rate for learning rate during regrowing. |
| `--start_lr_pruning`               | float   | No       | N/A                                        | Starting learning rate for pruning. |
| `--end_lr_pruning`                 | float   | No       | N/A                                        | Ending learning rate for pruning. |
| `--reset_lr_pruning`               | float   | No       | N/A                                        | Reset learning rate after pruning. |
| `--end_lr_regrowth`                | float   | No       | N/A                                        | Ending learning rate for regrowth. |
| `--reset_lr_flow_params_scaler`    | int     | No       | N/A                                        | Scaler for resetting learning rate flow parameters. |
| `--weight_decay`                   | float   | No       | N/A                                        | Weight decay parameter. |


You can either use one of the stable versions in our code for training the network or provide your own hyperparameters
For reproducing the results in the tables,
we recommend running each network with stable_version2, although stable_version1 works too. Make sure to
have the proper baselines preloaded


**Note that calculations in the method adjust the input sparsity such that after regrowth, it will reach the target sparsity. So you will initally reach sparsities lower than mentioed in the CLI**

To run ResNet50 CIFAR10 as an example we can use the following.
```sh
python entry.py  --model resnet50 --dataset cifar10  --training_type pruned  --target_sparsity 90  --version stable_version1
```

If you want to override parameters within versions you can use
```sh
python entry.py  --model resnet50 --dataset cifar10  --training_type pruned  --target_sparsity 90  --version stable_version1 --weight_decay 5e-4
```

And if you want to mention all you parameters from scratch, simply do not mention the version and override all arguments
