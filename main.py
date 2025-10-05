import argparse
import sys
from src.infrastructure.constants import INITIAL_LR 
from src.infrastructure.others import TrainingConfigsNPLHIMP 
from src.resnet50_cifar10.train_NPLH_IMP_resnet50_cifar10 import train_resnet50_cifar10_IMP
from src.vgg19_cifar100.train_NPLH_IMP_vgg19_cifar100 import train_vgg19_cifar100_IMP
from src.resnet50_imagenet1k.train_NPLH_IMP_resnet50_imagenet import train_resnet50_imagenet_NPLH_IMP
from src.resnet50_cifar10.train_sparsity_curves_adam import generate_cifar10_resnet50_adam_sparsity_curve
from src.resnet50_cifar10.train_sparsity_curves_sgd import run_cifar10_resnet50_sgd_sparsity_curve

def traing_r50c10_IMP(): 
   defaults: TrainingConfigsNPLHIMP = {
      "training_end": 500,
      "start_lr_pruning": INITIAL_LR / 10,
      "end_lr_pruning": INITIAL_LR / 10,
      "weight_decay": 5e-4,
      "target_sparsity": 99.975,
      "resume": "resnet50_cifar10_accuracy94.91%", 
   }

   train_resnet50_cifar10_IMP(defaults)

def train_vgg19_c100_IMP(): 
   defaults: TrainingConfigsNPLHIMP = {
      "training_end": 500,
      "start_lr_pruning": INITIAL_LR / 10,
      "end_lr_pruning": INITIAL_LR / 10,
      "weight_decay": 5e-4,
      "target_sparsity": 99.95,
      "resume": "vgg19_cifar100_accuracy72.9%", 
   }

   train_vgg19_cifar100_IMP(defaults)


if __name__ == "__main__":
   generate_cifar10_resnet50_adam_sparsity_curve(arg=2, power_start=1, power_end=1)
   run_cifar10_resnet50_sgd_sparsity_curve(arg=2, power_start=1, power_end=1)
   pass 