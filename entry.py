from src.resnet18_cifar10.run_existing_resnet18_cifar10 import run_resnet18_cifar10_existing_model
from src.infrastructure.constants import PRUNED_MODELS_PATH
from src.resnet18_cifar10.train_pruned_resnet18_cifar10_adam import train_resnet18_cifar10_sparse_model_adam
from src.resnet18_cifar10.train_scratch_resnet18_cifar10 import train_resnet18_cifar10_from_scratch
from src.resnet18_cifar100.train_pruned_resnet18_cifar100 import train_resnet18_cifar100_sparse_model
from src.resnet18_cifar100.train_scratch_resnet18_cifar100 import train_resnet18_cifar100_from_scratch
from src.resnet50_cifar10.train_pruned_resnet50_cifar10 import train_resnet50_cifar10_sparse_model
from src.resnet50_cifar10.train_scratch_resnet50_cifar10 import train_resnet50_cifar10_from_scratch
from src.resnet50_cifar100.train_pruned_resnet50_cifar100 import train_resnet50_cifar100_sparse_model
from src.resnet50_cifar100.train_scratch_resnet50_cifar100 import train_resnet50_cifar100_from_scratch
from src.vgg19_cifar10.train_pruned_vgg19_cifar10 import train_vgg19_cifar10_sparse_model
from src.vgg19_cifar10.train_scratch_vgg19_cifar10 import train_vgg19_cifar10_from_scratch
from src.vgg19_cifar100.train_scratch_vgg19_cifar100 import train_vgg19_cifar100_from_scratch
from src.resnet50_imagenet1k.train_pruned_resnet50_imagenet import train_imagenet_resnet50_sparse_model
import os

os.environ['HF_HOME'] = '/home/developer/workspace/not_ok/cache'
os.environ['HF_CACHE'] = '/home/developer/workspace/not_ok/cache'

if __name__ == '__main__':
    #train_resnet50_cifar100_from_scratch()
    # train_vgg19_cifar100_from_scratch()
    #
    # train_resnet18_cifar10_from_scratch()
    # train_resnet50_cifar10_from_scratch()
    # train_vgg19_cifar10_from_scratch()
    train_imagenet_resnet50_sparse_model()
    pass



# # import os
# # from huggingface_hub import login
# # from datasets import load_dataset

# # # Set a custom cache directory
# # os.environ["HF_HOME"] = "/home/developer/workspace/not_ok/cache"

# # # Log in with your Hugging Face token
# # login()

# # # Load training and validation datasets
# # train_dataset = load_dataset(
# #     "ILSVRC/imagenet-1k",
# #     split="train",
# #     cache_dir="/home/developer/workspace/Hyperflows/data/imagenet"
# # )
# # val_dataset = load_dataset(
# #     "ILSVRC/imagenet-1k",
# #     split="validation",
# #     cache_dir="/home/developer/workspace/Hyperflows/data/imagenet"
# # )

# # print("Training dataset size:", len(train_dataset))
# # print("Validation dataset size:", len(val_dataset))
# import torch
# print("CUDA available:", torch.cuda.is_available())
# print("Number of CUDA devices:", torch.cuda.device_count())
