import os

os.environ["HF_DATASETS_CACHE"] = "/home/developer/workspace/AntonioWork/old_versions/not_ok/data/imagenet"
os.environ["HF_HOME"] = "/home/developer/workspace/AntonioWork/old_versions/not_ok/data/os.environ.get("HF_TOKEN")"
os.environ["HF_MODULES_CACHE"] = "/home/developer/workspace/AntonioWork/old_versions/not_ok/data/os.environ.get("HF_TOKEN")/modules"

from src.vit.prune_vit_cifar100 import train_vit_cifar100_sparse_model
from src.vit.prune_vit_imagenet1k import train_vit_imagenet_sparse_model
import torch
from src.infrastructure.others import TrainingConfigsWithResume
from huggingface_hub import login
from datasets import load_dataset, DownloadMode, DownloadConfig
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

def vit_cifar100_sparsity_experiment(target_sparsity: float):
    INITIAL_LR = 0.1
    defaults: TrainingConfigsWithResume = {
        "pruning_end": 100,
        "regrowing_end": 160,
        "lr_flow_params_decay_regrowing": 0.75,
        "start_lr_pruning": INITIAL_LR,
        "end_lr_pruning": INITIAL_LR / 30,
        "reset_lr_pruning": INITIAL_LR / 100,
        "end_lr_regrowth": INITIAL_LR / 1000,
        "reset_lr_flow_params_scaler": 5,
        "weight_decay": 5e-4,
        "target_sparsity": target_sparsity,
        "notes": f"VIT CIFAR-100 {target_sparsity}% target sparsity",
    }
    train_vit_cifar100_sparse_model(defaults)


def vit_imagenet_sparsity_experiment(final_sparsity: float):
    # Allow passing either a fraction (0-1) representing remaining density
    # or a percentage (0-100) representing desired sparsity.
    if final_sparsity <= 1:
        # treat as fraction of remaining weights (e.g. 0.1 => 10% remaining => 90% sparsity)
        target_percent = 100 - final_sparsity * 100
    else:
        # treat as percentage sparsity already (e.g. 90 => 90% sparsity)
        target_percent = final_sparsity

    defaults: TrainingConfigsWithResume = {
        "pruning_end": 200,
        "regrowing_end": 300,
        "target_sparsity": target_percent,
        "lr_flow_params_decay_regrowing": 0.55,
        "start_lr_pruning": 0.0005,
        "end_lr_pruning": 0.00005,
        "reset_lr_pruning": 0.0005,
        "end_lr_regrowth": 0.00001,
        "reset_lr_flow_params_scaler": 3,
        "weight_decay": 1e-4,
        "notes": f"Running imagenet1k aiming for ~{target_percent}% sparsity (remaining {100 - target_percent}%)",
    }
    print("Starting ViT ImageNet sparsity experiment with the following target sparsity:")
    # print configs
    for key, value in defaults.items():
        print(f"  {key}: {value}")
    train_vit_imagenet_sparse_model(defaults)


if __name__ == "__main__":
#     dataset = load_dataset(
#     "ILSVRC/imagenet-1k",
#     cache_dir=os.environ["HF_DATASETS_CACHE"],
#     download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS,  # reuse any shard that exists
#     revision="07900defe1ccf3404ea7e5e876a64ca41192f6c07406044771544ef1505831e8",
#     download_config=DownloadConfig(local_files_only=False, resume_download=True),

# )
    vit_imagenet_sparsity_experiment(final_sparsity=0.1)
    # vit_cifar100_sparsity_experiment(target_sparsity=0)   

