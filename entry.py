import argparse
import sys

from src.resnet50_cifar100.train_pruned_resnet50_cifar100_steplr import train_resnet50_cifar100_sparse_model_steplr


def main():
    train_resnet50_cifar100_sparse_model_steplr({})

if __name__ == "__main__":
    main()