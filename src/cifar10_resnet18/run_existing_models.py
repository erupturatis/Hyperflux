import torchvision.models as models
import torch
from src.cifar10_resnet18.model_attributes import RESNET18_CIFAR10_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING
from src.cifar10_resnet18.model_class import ModelBaseResnet18
from src.infrastructure.dataset_context.dataset_context import DatasetSmallContext, DatasetSmallType, \
    dataset_context_configs_cifar10
from src.infrastructure.others import prefix_path_with_root, get_device, get_sparsity
from torch import nn

dataset_context: DatasetSmallContext

def test(model: ModelBaseResnet18, dataset_context: DatasetSmallContext, epoch):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    test_loss = 0
    correct = 0

    with torch.no_grad():
        while dataset_context.any_data_testing_available():
            data, target = dataset_context.get_testing_data_and_labels()

            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    total_data_len = dataset_context.get_data_testing_length()
    test_loss /= total_data_len
    accuracy = 100.0 * correct / total_data_len

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{total_data_len} ({accuracy:.0f}%)"
    )

def run_cifar10_resnet18_existing_model(model_name:str, folder: str):
    global dataset_context

    filepath = folder + '/' + model_name
    filepath = prefix_path_with_root(filepath)
    state_dict = torch.load(filepath)

    model = models.resnet18()

    # change first and last layer to match cifar10 architecture
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, 10)

    model.load_state_dict(state_dict)
    model = model.to(get_device())

    dataset_context = DatasetSmallContext(dataset=DatasetSmallType.CIFAR10, configs=dataset_context_configs_cifar10())

    num_epochs = 10
    for epoch in range(1, num_epochs):
        dataset_context.init_data_split()
        test(model, dataset_context, epoch)


