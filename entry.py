from src.resnet18_cifar10.run_existing_resnet18_cifar10 import run_resnet18_cifar10_existing_model
from src.infrastructure.constants import PRUNED_MODELS_PATH

if __name__ == '__main__':
    # train_cifar10_resnet18_sparse_model_adam()
    run_resnet18_cifar10_existing_model(folder=PRUNED_MODELS_PATH, model_name="resnet18c2")

    pass
