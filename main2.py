import argparse
import sys
from src.infrastructure.constants import INITIAL_LR 
from src.infrastructure.others import TrainingConfigsNPLHIMP 
from src.resnet50_cifar10.train_NPLH_IMP_resnet50_cifar10 import train_resnet50_cifar10_IMP
from src.vgg19_cifar100.train_NPLH_IMP_vgg19_cifar100 import train_vgg19_cifar100_IMP
from src.resnet50_cifar10.train_NPLH_L0_resnet50_cifar10 import train_resnet50_cifar10_L0_cont_registering

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


def resnet50_cifar10_NPLH_L0():
   train_resnet50_cifar10_L0_cont_registering({ 
      "pruning_end": 1000,
      "exponent_start":-10,
      "exponent_end":10,
      "base":2, 
      "epochs_raise":50,
      "learning_rate": INITIAL_LR / 10,
      "weight_decay": 5e-4,
      "resume": "resnet50_cifar10_accuracy94.91%", 
      "notes": f""
   })

if __name__ == "__main__":
   resnet50_cifar10_NPLH_L0()