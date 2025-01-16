import torchvision.models as models
import torch
from src.resnet18_cifar10.resnet18_cifar10_attributes import RESNET18_CIFAR10_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING
from src.common_files_experiments.test_existing_model import test_existing_model
from src.infrastructure.dataset_context.dataset_context import DatasetSmallContext, DatasetSmallType, \
    dataset_context_configs_cifar10
from src.infrastructure.others import prefix_path_with_root, get_device, get_sparsity
from src.resnet50_cifar10.resnet50_cifar10_attributes import RESNET50_CIFAR10_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING

def run_resnet50_cifar10_existing_model(model_name:str, folder: str):
    dataset_context: DatasetSmallContext

    filepath = folder + '/' + model_name
    filepath = prefix_path_with_root(filepath)
    state_dict = torch.load(filepath)

    model = models.resnet50()

    # change first and last layer to match cifar10 architecture
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, 10)

    model.load_state_dict(state_dict)
    model = model.to(get_device())
    get_sparsity(
        model=model,
        standard_to_custom_attributes=RESNET50_CIFAR10_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING
    )

    dataset_context = DatasetSmallContext(dataset=DatasetSmallType.CIFAR10, configs=dataset_context_configs_cifar10())

    num_epochs = 10
    for epoch in range(1, num_epochs):
        dataset_context.init_data_split()
        test_existing_model(model, dataset_context)


