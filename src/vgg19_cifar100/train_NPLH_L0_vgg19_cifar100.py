import torch

from src.infrastructure.stages_context.stages_context import StagesContextPrunedTrain, StagesContextPrunedTrainArgs
from src.infrastructure.training_context.training_context import TrainingContextPrunedTrain, \
    TrainingContextPrunedTrainArgs
from src.vgg19_cifar100.vgg19_cifar100_class import VGG19Cifar100
from src.common_files_experiments.train_pruned_commons import train_mixed_pruned, test_pruned
from src.infrastructure.configs_layers import configs_layers_initialization_all_kaiming_sqrt5, \
    configs_layers_initialization_all_kaiming_relu
from src.infrastructure.constants import config_adam_setup, get_lr_flow_params_reset, get_lr_flow_params, \
    PRUNED_MODELS_PATH, BASELINE_MODELS_PATH
from src.infrastructure.dataset_context.dataset_context import DatasetSmallContext, DatasetSmallType, dataset_context_configs_cifar100
from src.infrastructure.training_display import TrainingDisplay, ArgsTrainingDisplay
from src.infrastructure.layers import ConfigsNetworkMasksImportance
from src.infrastructure.others import get_device, get_custom_model_sparsity_percent, get_random_id, \
    TrainingConfigsWithResume
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from src.infrastructure.schedulers import PressureSchedulerPolicy1
from src.infrastructure.training_common import get_model_flow_params_and_weights_params
from src.infrastructure.wandb_functions import wandb_initalize, wandb_finish, Experiment, Tags
from src.infrastructure.stages_context.stages_context import StagesContextNPLHTrain, StagesContextNPLHTrainArgs
from src.infrastructure.training_context.training_context import TrainingContextNPLHL0, TrainingContextNPLHL0Args
from src.infrastructure.schedulers import PressureSchedulerPolicyMeasurements
from src.infrastructure.others import TrainingConfigsNPLHL0
from src.infrastructure.read_write import save_dict_to_csv

def initialize_model():
    global MODEL, training_configs
    configs_network_masks = ConfigsNetworkMasksImportance(
        mask_pruning_enabled=True,
        weights_training_enabled=True,
    )
    MODEL = VGG19Cifar100(configs_network_masks).to(get_device())
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
            average_losses_names=["Loss Data", "Loss Remaining Weights"],
            model=MODEL,
            batch_print_rate=BATCH_PRINT_RATE,
            get_epoch= get_epoch
        )
    )

def initialize_dataset_context():
    global dataset_context
    dataset_context = DatasetSmallContext(dataset=DatasetSmallType.CIFAR100, configs=dataset_context_configs_cifar100())


def initialize_training_context():
    global training_context

    lr_weights_finetuning = training_configs["learning_rate"]
    lr_flow_params = get_lr_flow_params()

    weight_bias_params, flow_params = get_model_flow_params_and_weights_params(MODEL)
    optimizer_weights = torch.optim.SGD(lr=lr_weights_finetuning, params= weight_bias_params, momentum=0.9, weight_decay=training_configs["weight_decay"])
    optimizer_flow_mask = torch.optim.Adam(lr=lr_flow_params, params=flow_params, weight_decay=0)

    training_context = TrainingContextNPLHL0(
        TrainingContextNPLHL0Args(
            optimizer_weights=optimizer_weights,
            optimizer_flow_mask=optimizer_flow_mask, 
            l0_gamma_scaler=0
        )
    )

def initialize_stages_context():
    global stages_context, training_context

    pruning_end = training_configs["pruning_end"]
    pruning_scheduler = PressureSchedulerPolicyMeasurements(
        exponent_start=training_configs["exponent_start"],
        exponent_end=training_configs["exponent_end"],
        base=training_configs["base"], 
        epochs_raise=training_configs["epochs_raise"]
    )

    stages_context = StagesContextNPLHTrain(
        StagesContextNPLHTrainArgs(
            pruning_epoch_end=pruning_end,
            scheduler_gamma=pruning_scheduler,
        )
    )

MODEL: VGG19Cifar100
training_context: TrainingContextNPLHL0
dataset_context: DatasetSmallContext
stages_context: StagesContextNPLHTrain
training_display: TrainingDisplay
epoch_global: int = 0
BATCH_PRINT_RATE = 100

training_configs: TrainingConfigsNPLHL0

def train_vgg19_cifar100_sparse_model(sparsity_configs_aux: TrainingConfigsNPLHL0):
    global MODEL, epoch_global, training_configs
    sparsity_configs = sparsity_configs_aux
    training_configs = sparsity_configs_aux

    configs_layers_initialization_all_kaiming_relu()
    config_adam_setup()

    initialize_model()
    initialize_training_context()
    initialize_stages_context()
    wandb_initalize(Experiment.VGG19CIFAR100, type=Tags.TRAIN_PRUNING, configs=sparsity_configs,other_tags=["ADAM"])
    initialize_dataset_context()
    initalize_training_display()

    acc = 0
    epochs = []
    remainings = []
    saliencies = []
    accuracies = []

    id = get_random_id()
    for epoch in range(1, training_configs["pruning_end"] + 1):
        epoch_global = epoch
        dataset_context.init_data_split()
        train_mixed_pruned(
            dataset_context=dataset_context,
            training_context=training_context,
            model=MODEL,
            training_display=training_display,
        )
        acc = test_pruned(
            dataset_context=dataset_context,
            model=MODEL,
            epoch=get_epoch()
        )
        stages_context.args.scheduler_gamma.acc = acc

        stages_context.update_context(epoch_global, get_custom_model_sparsity_percent(MODEL))
        stages_context.step(training_context)


        if epoch % training_configs["epochs_raise"] == 0: 
            epochs.append(epoch)
            remainings.append(get_custom_model_sparsity_percent(MODEL))
            saliencies.append(training_context.params.l0_gamma_scaler)
            accuracies.append(acc)

            print(epochs)
            print(remainings)
            print(saliencies)
            print(accuracies)
            pass
  
        # values = get_populated_values()
        save_dict_to_csv({
            "Saliency": saliencies, 
            "Remaining": remainings,
            "Accuracy": accuracies,
            "Epochs": epochs
        },
        filename="L0vgg19c100.csv" 
        )

    MODEL.save(
        name= f"vgg19_cifar100_sparsity{get_custom_model_sparsity_percent(MODEL)}_acc{acc}_{id}",
        folder= PRUNED_MODELS_PATH
    )
    print("Training complete")
    wandb_finish()
