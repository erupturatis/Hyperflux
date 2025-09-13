import torch
from src.common_files_experiments.train_model_scratch_commons import train_mixed_baseline, test_baseline
from src.common_files_experiments.train_pruned_commons import train_mixed_pruned, test_pruned
from src.infrastructure.configs_layers import configs_layers_initialization_all_kaiming_sqrt5, \
    configs_layers_initialization_all_kaiming_relu
from src.infrastructure.constants import config_adam_setup, get_lr_flow_params_reset, get_lr_flow_params, \
    PRUNED_MODELS_PATH, BASELINE_MODELS_PATH
from src.infrastructure.dataset_context.dataset_context import DatasetSmallContext, DatasetSmallType, dataset_context_configs_cifar10
from src.infrastructure.stages_context.stages_context import \
    StagesContextBaselineTrain, StagesContextBaselineTrainArgs
from src.infrastructure.training_context.training_context import \
    TrainingContextBaselineTrain, TrainingContextBaselineTrainArgs
from src.infrastructure.training_display import TrainingDisplay, ArgsTrainingDisplay
from src.infrastructure.layers import ConfigsNetworkMasksImportance
from src.infrastructure.others import get_device, get_custom_model_sparsity_percent, get_random_id
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from src.infrastructure.schedulers import PressureSchedulerPolicy1
from src.infrastructure.training_common import get_model_weights_params
from src.infrastructure.wandb_functions import wandb_initalize, wandb_finish, Experiment, Tags
from src.resnet50_cifar10.resnet50_cifar10_class import Resnet50Cifar10
from src.infrastructure.others import get_device, TrainingConfigsIMP
from src.infrastructure.layers import prune_model_globally, calculate_pruning_epochs
from src.infrastructure.read_write import save_dict_to_csv

def initialize_model():
    global MODEL
    configs_network_masks = ConfigsNetworkMasksImportance(
        mask_pruning_enabled=False,
        weights_training_enabled=True,
    )

    MODEL = Resnet50Cifar10(configs_network_masks).to(get_device())
    if "resume" in training_configs:
        MODEL.load(training_configs["resume"], BASELINE_MODELS_PATH)

def get_epoch() -> int:
    global epoch_global
    return epoch_global

def initalize_training_display():
    global training_display
    training_display = TrainingDisplay(
        args=ArgsTrainingDisplay(
            dataset_context=dataset_context,
            average_losses_names=["Loss Data"],
            model=MODEL,
            batch_print_rate=BATCH_PRINT_RATE,
            get_epoch= get_epoch
        )
    )

def initialize_dataset_context():
    global dataset_context
    dataset_context = DatasetSmallContext(dataset=DatasetSmallType.CIFAR10, configs=dataset_context_configs_cifar10())

def initialize_training_context():
    global training_context

    lr = training_configs["start_lr_pruning"]
    weight_bias_params = get_model_weights_params(MODEL)
    optimizer_weights = torch.optim.SGD(lr=lr, params=weight_bias_params, momentum=0.9, weight_decay=training_configs["weight_decay"], nesterov=True)

    training_context = TrainingContextBaselineTrain(
        TrainingContextBaselineTrainArgs(
            optimizer_weights=optimizer_weights,
        )
    )

def initialize_stages_context():
    global stages_context, training_context

    training_end = training_configs["training_end"]
    scheduler_weights_lr_during_pruning = CosineAnnealingLR(training_context.get_optimizer_weights(), T_max=training_end, eta_min=training_configs["end_lr_pruning"])

    stages_context = StagesContextBaselineTrain(
        StagesContextBaselineTrainArgs(
            training_end=training_end,
            scheduler_weights_lr_during_training=scheduler_weights_lr_during_pruning,
        )
    )

MODEL: Resnet50Cifar10
training_context: TrainingContextBaselineTrain
dataset_context: DatasetSmallContext
stages_context: StagesContextBaselineTrain
training_display: TrainingDisplay
epoch_global: int = 0
BATCH_PRINT_RATE = 100

training_configs: TrainingConfigsIMP

def train_resnet50_cifar10_IMP(conf: TrainingConfigsIMP):
    global MODEL, epoch_global, training_configs

    training_configs = conf
    configs_layers_initialization_all_kaiming_relu()

    initialize_model()
    initialize_training_context()
    initialize_stages_context()
    wandb_initalize(experiment=Experiment.RESNET50CIFAR10, type=Tags.IMP)
    initialize_dataset_context()
    initalize_training_display()

    pruning_rate = 0.1
    epochs_to_prune = []
    epochs_to_prune = calculate_pruning_epochs(
        target_sparsity=training_configs["target_sparsity"]/100, 
        pruning_rate=pruning_rate, 
        total_epochs=training_configs["training_end"], 
        start_epoch=1
    )
    print("EPOCHS TO PRUNE", epochs_to_prune)

    acc = 0
    thresholds = []
    remaining_params = []
    pruned_epochs = []
    accuracies = []

    for epoch in range(1, training_configs["training_end"] + 1):
        epoch_global = epoch
        dataset_context.init_data_split()

        train_mixed_baseline(
            model=MODEL,
            dataset_context=dataset_context,
            training_context=training_context,
            training_display=training_display,
        )
        acc = test_baseline(
            model=MODEL,
            dataset_context=dataset_context,
            epoch=epoch,
        )

        stages_context.update_context(epoch_global)
        stages_context.step(training_context)
        
        if epoch in epochs_to_prune: 
            val = prune_model_globally(MODEL, pruning_rate)
            rem = get_custom_model_sparsity_percent(MODEL)
            thresholds.append(val)
            remaining_params.append(rem)
            pruned_epochs.append(epoch)
            accuracies.append(acc)

            print(thresholds)
            print(remaining_params)
            print(pruned_epochs)
            print(accuracies)
    
    save_dict_to_csv({
        "Epoch": epochs_to_prune, 
        "SaliencyIMP": thresholds, 
        "RemainingParams": remaining_params,
        "Accuracy": accuracies
    },
    filename="impr50c10.csv" 
    )

    # MODEL.save(
    #     name=f"resnet50_cifar10_accuracy{acc}%_{get_random_id()}",
    #     folder=BASELINE_MODELS_PATH
    # )

    print("Training complete")
    wandb_finish()
