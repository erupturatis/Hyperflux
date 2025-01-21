import torch
import torch.nn as nn
from src.infrastructure.dataset_context.dataset_context import DatasetContextAbstract
from src.infrastructure.layers import LayerComposite
from src.infrastructure.others import get_model_sparsity_percent
from torch.amp import GradScaler, autocast
from src.infrastructure.training_context.training_context import TrainingContextPrunedTrain
from src.infrastructure.training_display import TrainingDisplay
from src.infrastructure.wandb_functions import wandb_snapshot

def train_mixed_pruned_with_decay(model: LayerComposite, dataset_context: DatasetContextAbstract, training_context: TrainingContextPrunedTrain, training_display: TrainingDisplay):
    model.train()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer_weights = training_context.get_optimizer_weights()
    optimizer_pruning = training_context.get_optimizer_flow_mask()

    scaler = GradScaler('cuda')

    while dataset_context.any_data_training_available():
        data, target = dataset_context.get_training_data_and_labels()

        optimizer_weights.zero_grad()
        optimizer_pruning.zero_grad()

        with autocast('cuda'):
            output = model(data)
            loss_remaining_weights = model.get_remaining_parameters_loss() * training_context.params.l0_gamma_scaler
            loss_weight_decay = model.get_weight_decay()

            loss_data = criterion(output, target)
            loss = loss_remaining_weights + loss_data + loss_weight_decay

        scaler.scale(loss).backward()
        scaler.step(optimizer_weights)
        scaler.step(optimizer_pruning)
        scaler.update()

        training_display.record_losses([loss_data.item(), loss_remaining_weights.item()])


def train_mixed_pruned(model: LayerComposite, dataset_context: DatasetContextAbstract, training_context: TrainingContextPrunedTrain, training_display: TrainingDisplay):
    model.train()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer_weights = training_context.get_optimizer_weights()
    optimizer_pruning = training_context.get_optimizer_flow_mask()

    scaler = GradScaler('cuda')

    while dataset_context.any_data_training_available():
        data, target = dataset_context.get_training_data_and_labels()

        optimizer_weights.zero_grad()
        optimizer_pruning.zero_grad()

        with autocast('cuda'):
            output = model(data)
            loss_remaining_weights = model.get_remaining_parameters_loss() * training_context.params.l0_gamma_scaler
            loss_data = criterion(output, target)
            loss = loss_remaining_weights + loss_data

        scaler.scale(loss).backward()
        scaler.step(optimizer_weights)
        scaler.step(optimizer_pruning)
        scaler.update()

        training_display.record_losses([loss_data.item(), loss_remaining_weights.item()])


def test_pruned(model: nn.Module, dataset_context: DatasetContextAbstract, epoch: int):
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

    remain_percent = get_model_sparsity_percent(model)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{total_data_len} ({accuracy:.0f}%)"
    )
    print(
        f"Remaining parameters: {remain_percent:.2f}%"
    )
    wandb_snapshot(epoch=epoch, accuracy=accuracy, test_loss=test_loss, sparsity=remain_percent)
    return accuracy
