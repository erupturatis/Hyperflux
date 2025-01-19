import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.infrastructure.others import get_device


mean_cifar100, std_cifar100 = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
mean_cifar10, std_cifar10 = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
mean_mnist, std_mnist = (0.1307,), (0.3081,)

def cifar100_preprocess() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean_cifar100, std=std_cifar100)
    ])

    trainset = datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    testset = datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=len(trainset),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        testset,
        batch_size=len(testset),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    train_data, train_labels = next(iter(train_loader))
    train_data = train_data.to(get_device())
    train_labels = train_labels.to(get_device())

    test_data, test_labels = next(iter(test_loader))
    test_data = test_data.to(get_device())
    test_labels = test_labels.to(get_device())

    return train_data, train_labels, test_data, test_labels



def cifar10_preprocess() -> tuple[ torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    trainset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean_cifar10, std_cifar10),
            ]
        ),
    )

    testset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean_cifar10, std_cifar10),
            ]
        ),
    )

    train_loader = DataLoader(
        trainset,
        batch_size=len(trainset),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        testset,
        batch_size=len(testset),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    train_data, train_labels = next(iter(train_loader))
    train_data = train_data.to(get_device())
    train_labels = train_labels.to(get_device())

    test_data, test_labels = next(iter(test_loader))
    test_data = test_data.to(get_device())
    test_labels = test_labels.to(get_device())

    return train_data, train_labels, test_data, test_labels

# mean_mnist, std_mnist = (0.1307,), (0.3081,)

def mnist_preprocess() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_mnist, std_mnist),
    ])

    train_dataset = datasets.MNIST(root='./data', train=True,
                                   download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    test_dataset = datasets.MNIST(root='./data', train=False,
                                  download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    train_data, train_labels = next(iter(train_loader))
    train_data = train_data.to(get_device())
    train_labels = train_labels.to(get_device())

    test_data, test_labels = next(iter(test_loader))
    test_data = test_data.to(get_device())
    test_labels = test_labels.to(get_device())

    return train_data, train_labels, test_data, test_labels
