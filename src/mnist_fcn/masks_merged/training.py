import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.utils import get_device
from .network_mnist_merged import ModelMnistFNNMergedMask
from src.mnist_fcn.masks_merged.others import ConfigsNetworkMasksMerged
from src.constants import WEIGHTS_ATTR, BIAS_ATTR, MASK_MERGED_ATTR

exp = 0


def balancer_parameters(network_loss: float, regularization_loss: float, scale:float = 1, ratio: float = 1) -> tuple[float, float]:
    """
    Balances the network losses in the desired ratio
    :param network_loss: ...
    :param regularization_loss: ...
    :param scale: represents the value of the final network loss
    :param ratio: represents the ratio: regularization_loss / network_loss
    :return:
    """
    a = scale / network_loss
    b = (a * ratio * network_loss) / regularization_loss
    return a, b

def train(model: ModelMnistFNNMergedMask, train_loader, optimizer, epoch):
    global exp
    model.train()
    criterion = nn.CrossEntropyLoss()
    device = get_device()

    avg_loss_masks = 0
    avg_loss_images = 0

    # if epoch == 1:
    #     model.fc1.set_mask(True)
    # if epoch == 3:
    #     model.fc2.set_mask(True)
    #     model.fc1.disable_mask_grad()
    # if epoch == 5:
    #     model.fc3.set_mask(True)
    #     model.fc2.disable_mask_grad()

    # if epoch == 3:
    #     model.fc3.set_mask(True)
    # if epoch == 6:
    #     model.fc2.set_mask(True)
    #     model.fc3.disable_mask_grad()
    # if epoch == 9:
    #     model.fc1.set_mask(True)
    #     model.fc2.disable_mask_grad()

    if epoch >= 4:
        model.enable_weights_training()

    for batch_idx, (data, target) in enumerate(train_loader):
        accumulated_loss = torch.tensor(0.0).to(device)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss_masks = model.get_prune_regularization_loss()
        loss_masks *= 0 if epoch < 4 else epoch ** 2

        # loss_masks *= 15 # meh, 0.5 with 96+%
        # loss_masks *= 5 # optimal, beat paper, 0.7, 97+%
        # loss_masks *= (epoch) # optimal, beat paper, 0.6, 97+%, converges slow
        # loss_masks *= 0 if epoch < 4 else 5 # 0.85, 98%

        # a,b = balancer_parameters(loss.item(), loss_masks.item(), scale=0.1, ratio=(epoch//2))
        # loss *= a
        # loss_masks *= b

        accumulated_loss += loss
        accumulated_loss += loss_masks

        avg_loss_masks += loss_masks.item()
        avg_loss_images += loss.item()

        accumulated_loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}]')
            print(f'Loss masks: {loss_masks.item():.6f} Loss images: {loss.item():.6f}')

            percentages_array = model.get_masked_percentages_per_layer()
            minus_one_arr = percentages_array[0]
            zero_arr = percentages_array[1]
            one_arr = percentages_array[2]

            for i in range(len(minus_one_arr)):
                print(f'Masks percentages: -1: {minus_one_arr[i]:.2f}%, 0: {zero_arr[i]:.2f}%, 1: {one_arr[i]:.2f}%')

            # overall pruning
            percentages = model.get_masked_percentages()
            minus_one = percentages[0]
            zero = percentages[1]
            one = percentages[2]

            print(f'Masks percentages: -1: {minus_one:.2f}%, 0: {zero:.2f}%, 1: {one:.2f}%')



    avg_loss_masks /= len(train_loader.dataset)
    avg_loss_images /= len(train_loader.dataset)

    # if(avg_loss_masks > avg_loss_images):
    #     exp += 1
    #     print("EXPONENT INCREASED")

def test(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(get_device()), target.to(get_device())
            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)      # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    global exp
    if accuracy <= 97.5:
        # exp += 1
        # print("EXPONENT INCREASED")
        pass

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({100. * correct / len(test_loader.dataset):.0f}%)\n')



def run_mnist_merged_masks():
    # Define transformations for the training and testing data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    batch_size = 128

    # Download and load the training data
    train_dataset = datasets.MNIST(root='./data', train=True,
                                   download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Download and load the test data
    test_dataset = datasets.MNIST(root='./data', train=False,
                                  download=True, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    configs_network_masks: ConfigsNetworkMasksMerged  = {
        'mask_merged_enabled': True,
        'weights_training_enabled': False,
    }
    # Instantiate the network, optimizer, etc.

    weight_bias_params = []
    mask_merged_params = []

    model = ModelMnistFNNMergedMask(configs_network_masks).to(get_device())

    for name, param in model.named_parameters():
        if WEIGHTS_ATTR in name or BIAS_ATTR in name:
            weight_bias_params.append(param)
        if MASK_MERGED_ATTR in name:
            mask_merged_params.append(param)

    lr_weight_bias = 0.01
    lr_custom_params = lr_weight_bias * 15

    optimizer = torch.optim.AdamW([
        {'params': weight_bias_params, 'lr': lr_weight_bias},
        {'params': mask_merged_params, 'lr': lr_custom_params},
    ])

    lambda_lr_weight_bias = lambda epoch: 0.5 ** (epoch // 2)
    lambda_lr_merged_params = lambda epoch: 1 ** (epoch // 2)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda_lr_weight_bias, lambda_lr_merged_params])

    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        # Toggle mask as needed
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)
        scheduler.step()

    print("Training complete")