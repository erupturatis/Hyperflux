import os

os.environ["HF_DATASETS_CACHE"] = "/home/developer/workspace/AntonioWork/old_versions/not_ok/data/imagenet"
os.environ["HF_HOME"] = "/home/developer/workspace/AntonioWork/old_versions/not_ok/data/hf_home"
os.environ["HF_MODULES_CACHE"] = "/home/developer/workspace/AntonioWork/old_versions/not_ok/data/hf_home/modules"

from src.vit.prune_vit_cifar100 import train_vit_cifar100_sparse_model
from src.vit.prune_vit_imagenet1k import train_vit_imagenet_sparse_model
from src.infrastructure.constants import INITIAL_LR 
import torch
from src.infrastructure.others import TrainingConfigsWithResume
from huggingface_hub import login
from datasets import load_dataset, DownloadMode, DownloadConfig
from src.resnet50_imagenet1k.run_existing_resnet50_imagenet import run_imagenet_resnet50_existing_model
from src.resnet50_imagenet1k.train_NPLH_IMP_resnet50_imagenet import train_resnet50_imagenet_NPLH_IMP


def resnet50_imagenet_run_existing(): 
    run_imagenet_resnet50_existing_model(
        model_name="baseline_imagenet.pth", 
        folder="networks_baseline"
    )

def resnet50_imagenet_NPLH_IMP(): 
   defaults: TrainingConfigsNPLHIMP = {
      "training_end": 200,
      "start_lr_pruning": INITIAL_LR / 10,
      "end_lr_pruning": INITIAL_LR / 10,
      "weight_decay": 5e-4,
      "target_sparsity": 99.75,
      "resume": "baseline_imagenet.pth", 
   }

   train_resnet50_imagenet_NPLH_IMP(defaults)


if __name__ == "__main__":
    resnet50_imagenet_NPLH_IMP()


