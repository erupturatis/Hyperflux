from enum import Enum
from typing import Tuple
import kornia.augmentation as K
from abc import ABC, abstractmethod
from src.infrastructure.dataset_context.data_preprocessing import cifar10_preprocess, mnist_preprocess
import torch
from src.infrastructure.others import get_device
import torch.nn as nn
from dataclasses import dataclass

class DatasetSmallType(Enum):
    CIFAR10 = 'cifar10'
    CIFAR100 = 'cifar100'
    MNIST = 'mnist'

@dataclass
class DatasetContextConfigs:
    batch_size: int
    augmentations: nn.Sequential

AUGMENTATIONS_CIFAR_10 = nn.Sequential(
    K.RandomCrop((32, 32), padding=4),
    K.RandomRotation(degrees=10.0),
    K.RandomHorizontalFlip(p=0.5),
).to(get_device())

BATCH_SIZE_CIFAR_10 = 128
BATCH_SIZE_MNIST = 128

def dataset_context_configs_cifar10() -> DatasetContextConfigs:
    return DatasetContextConfigs(batch_size=BATCH_SIZE_CIFAR_10, augmentations=AUGMENTATIONS_CIFAR_10)

def dataset_context_configs_mnist() -> DatasetContextConfigs:
    return DatasetContextConfigs(batch_size=BATCH_SIZE_MNIST, augmentations=None)

class DatasetContextAbstract(ABC):
    # TRAINING
    @abstractmethod
    def get_total_batches_training(self) -> int:
        pass

    @abstractmethod
    def get_batch_training_index(self) -> int:
        pass

    @abstractmethod
    def any_data_training_available(self) -> bool:
        pass

    @abstractmethod
    def get_data_training_length(self) -> int:
        pass

    # TESTING
    @abstractmethod
    def get_total_batches_testing(self) -> int:
        pass

    @abstractmethod
    def get_batch_testing_index(self) -> int:
        pass

    @abstractmethod
    def any_data_testing_available(self) -> bool:
        pass

    @abstractmethod
    def get_data_testing_length(self) -> int:
        pass

    @abstractmethod
    def get_batch_size(self) -> int:
        pass


class DatasetSmallContext(DatasetContextAbstract):
    def __init__(self, dataset: DatasetSmallType, configs: DatasetContextConfigs):
        """
        Loads the entire dataset on the GPU, since dataset is small
        """

        self.training_data_indices_iterator = None
        self.testing_data_indices_iterator = None
        self.dataset = dataset
        self.configs = configs

        self.batch_training_index = 0
        self.batch_test_index = 0

        if dataset == DatasetSmallType.CIFAR10:
            self.train_data, self.train_labels, self.test_data, self.test_labels = cifar10_preprocess()
        if dataset == DatasetSmallType.MNIST:
            self.train_data, self.train_labels, self.test_data, self.test_labels = mnist_preprocess()
        if dataset == DatasetSmallType.CIFAR100:
            pass

        self.total_training_batches = len(self.train_labels) // self.configs.batch_size
        self.total_test_batches = len(self.test_labels) // self.configs.batch_size

        if len(self.train_labels) % self.configs.batch_size != 0:
            self.total_training_batches += 1
        if len(self.test_labels) % self.configs.batch_size != 0:
            self.total_test_batches += 1

    def init_data_split(self):
        batch_size = self.configs.batch_size
        device = get_device()

        total_training_data_len = len(self.train_data)
        indices = torch.randperm(total_training_data_len, device=device)
        batch_indices = torch.split(indices, batch_size)
        self.training_data_indices_iterator = iter(enumerate(batch_indices))
        self.batch_training_index = 0

        total_test_data_len = len(self.test_data)
        indices = torch.randperm(total_test_data_len, device=device)
        batch_indices = torch.split(indices, batch_size)
        self.testing_data_indices_iterator = iter(enumerate(batch_indices))
        self.batch_test_index = 0

    # TRAINING
    def get_total_batches_training(self) -> int:
        return self.total_training_batches

    def get_batch_training_index(self) -> int:
        return self.batch_training_index

    def any_data_training_available(self) -> bool:
        return self.batch_training_index < self.total_training_batches

    def get_data_training_length(self) -> int:
        return len(self.train_labels)

    def get_training_data_and_labels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.batch_training_index += 1
        batch_idx, batch = next(self.training_data_indices_iterator)

        data = self.train_data[batch].to(get_device(), non_blocking=True)
        target = self.train_labels[batch].to(get_device(), non_blocking=True)

        if self.configs.augmentations is not None:
            data = self.configs.augmentations(data)

        return data, target

    # TESTING
    def get_total_batches_testing(self) -> int:
        return self.total_test_batches

    def get_batch_testing_index(self) -> int:
        return self.batch_test_index

    def any_data_testing_available(self) -> bool:
        return self.batch_test_index < self.total_test_batches

    def get_data_testing_length(self) -> int:
        return len(self.test_data)

    def get_batch_size(self) -> int:
        return self.configs.batch_size

    def get_testing_data_and_labels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.batch_test_index += 1
        batch_idx, batch = next(self.testing_data_indices_iterator)
        data = self.test_data[batch].to(get_device(), non_blocking=True)
        target = self.test_labels[batch].to(get_device(), non_blocking=True)

        return data, target



