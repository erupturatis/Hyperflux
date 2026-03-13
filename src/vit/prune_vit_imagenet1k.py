import torch
from src.common_files_experiments.train_pruned_commons import (
    train_mixed_pruned,
    test_pruned,
    train_mixed_pruned_imagenet,
    test_pruned_imagenet,
)
from src.infrastructure.stages_context.stages_context import (
    StagesContextPrunedTrain,
    StagesContextPrunedTrainArgs,
)
from src.infrastructure.training_context.training_context import (
    TrainingContextPrunedTrain,
    TrainingContextPrunedTrainArgs,
)
from src.infrastructure.configs_layers import (
    configs_layers_initialization_all_kaiming_sqrt5,
    configs_layers_initialization_all_kaiming_relu,
)
from src.infrastructure.constants import (
    config_adam_setup,
    get_lr_flow_params_reset,
    get_lr_flow_params,
    PRUNED_MODELS_PATH,
    BASELINE_MODELS_PATH,
)
from src.infrastructure.dataset_context.dataset_context import (
    DatasetSmallContext,
    DatasetSmallType,
    dataset_context_configs_cifar10,
    DatasetImageNetContext,
    DatasetImageNetContextConfigs,
)
from src.infrastructure.training_display import TrainingDisplay, ArgsTrainingDisplay
from src.infrastructure.layers import ConfigsNetworkMasksImportance
from src.infrastructure.others import (
    get_device,
    get_custom_model_sparsity_percent,
    TrainingConfigsWithResume,
)
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from src.infrastructure.schedulers import PressureSchedulerPolicy1
from src.infrastructure.training_common import (
    get_model_flow_params_and_weights_params,
    get_model_flow_params_and_weights_params_bn_separate,
)
from src.infrastructure.wandb_functions import (
    wandb_initalize,
    wandb_finish,
    Experiment,
    Tags,
)
from torch import nn
from src.vit.prunable_vit import VisionTransformerPrunable
import torchvision.models as models

MODEL: VisionTransformerPrunable
MODEL_MODULE: any
training_context: TrainingContextPrunedTrain
dataset_context: DatasetImageNetContext
stages_context: StagesContextPrunedTrain
training_display: TrainingDisplay
epoch_global: int = 0
BATCH_PRINT_RATE = 100
training_configs: TrainingConfigsWithResume


def initialize_model():
    global MODEL, MODEL_MODULE, training_configs

    configs_network_masks = ConfigsNetworkMasksImportance(
        mask_pruning_enabled=True,
        weights_training_enabled=True,
    )

    # Deit_Base/16
    # MODEL = VisionTransformerPrunable(
    #     configs_network_masks=configs_network_masks,
    #     img_size=224,
    #     patch_size=16,
    #     in_chans=3,
    #     num_classes=1000,
    #     embed_dim=768,  #ViT-Base/16
    #     depth=12,
    #     num_heads=12,
    #     mlp_ratio=4.0,
    #     qkv_bias=True,
    #     drop_rate=0.1,
    #     attn_drop_rate=0.1,
    #     drop_path_rate=0.1,
    # )
    # Deit_Tiny/16
    MODEL = VisionTransformerPrunable(
        configs_network_masks=configs_network_masks,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=192,   # was 768
        depth=12,
        num_heads=3,     # was 12
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        )
    MODEL.load_weights()
    

    print(f"Number of available CUDA devices: {torch.cuda.device_count()}")
    
    # Wrap with DataParallel BEFORE moving to device
    # Use all 3 GPUs to spread batch 512: 512/3 = ~170 per GPU (vs 256 on 2 GPUs)
    if torch.cuda.device_count() >= 3:
        MODEL = nn.DataParallel(MODEL, device_ids=[0, 1, 2])
        MODEL = MODEL.to('cuda:0')
        MODEL_MODULE = MODEL.module
    else:
        MODEL = MODEL.to(get_device())
        MODEL_MODULE = MODEL



def get_epoch() -> int:
    global epoch_global
    return epoch_global


def initalize_training_display():
    global training_display
    training_display = TrainingDisplay(
        args=ArgsTrainingDisplay(
            dataset_context=dataset_context,
            average_losses_names=["Loss Data", "Loss Remaining Weights"],
            model=MODEL_MODULE,
            batch_print_rate=BATCH_PRINT_RATE,
            get_epoch=get_epoch,
        )
    )


def initialize_dataset_context():
    global dataset_context
    configs = DatasetImageNetContextConfigs(
        batch_size=1024,
        use_mixup_cutmix = False,
    )
    dataset_context = DatasetImageNetContext(configs)


def initialize_training_context():
    global training_context

    lr_weights_finetuning = training_configs["start_lr_pruning"]
    lr_flow_params = get_lr_flow_params()

    weight_bias_params, flow_params = get_model_flow_params_and_weights_params(MODEL)

    # this matches typical ViT-B/ImageNet training recipes
    optimizer_weights = torch.optim.AdamW(
        lr=lr_weights_finetuning,
        params=weight_bias_params,
        weight_decay=training_configs["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    optimizer_flow_mask = torch.optim.Adam(
        lr=lr_flow_params, params=flow_params, weight_decay=0
    )

    training_context = TrainingContextPrunedTrain(
        TrainingContextPrunedTrainArgs(
            lr_weights_reset=training_configs["reset_lr_pruning"],
            lr_flow_params_reset=get_lr_flow_params()
            * training_configs["reset_lr_flow_params_scaler"],
            l0_gamma_scaler=0,
            optimizer_weights=optimizer_weights,
            optimizer_flow_mask=optimizer_flow_mask,
        )
    )


def initialize_stages_context():
    global stages_context, training_context

    pruning_end = training_configs["pruning_end"]
    regrowing_end = training_configs["regrowing_end"]
    regrowth_stage_length = regrowing_end - pruning_end

    pruning_scheduler = PressureSchedulerPolicy1(
        pressure_exponent_constant=1.5,
        sparsity_target=training_configs["target_sparsity"],
        epochs_target=pruning_end,
        step_size=0.15,
    )
    scheduler_decay_after_pruning = training_configs["lr_flow_params_decay_regrowing"]

    scheduler_weights_lr_during_pruning = CosineAnnealingLR(
        training_context.get_optimizer_weights(),
        T_max=pruning_end,
        eta_min=training_configs["end_lr_pruning"],
    )
    scheduler_weights_lr_during_regrowth = CosineAnnealingLR(
        training_context.get_optimizer_weights(),
        T_max=regrowth_stage_length,
        eta_min=training_configs["end_lr_regrowth"],
    )
    scheduler_flow_params_lr_during_regrowth = LambdaLR(
        training_context.get_optimizer_flow_mask(),
        lr_lambda=lambda iter: scheduler_decay_after_pruning**iter if iter < 50 else 0,
    )

    stages_context = StagesContextPrunedTrain(
        StagesContextPrunedTrainArgs(
            pruning_epoch_end=pruning_end,
            regrowth_epoch_end=regrowing_end,
            scheduler_gamma=pruning_scheduler,
            scheduler_weights_lr_during_pruning=scheduler_weights_lr_during_pruning,
            scheduler_flow_params_regrowth=scheduler_flow_params_lr_during_regrowth,
            scheduler_weights_lr_during_regrowth=scheduler_weights_lr_during_regrowth,
        )
    )


def train_vit_imagenet_sparse_model(sparsity_configs_aux: TrainingConfigsWithResume):
    global epoch_global, MODEL_MODULE, training_configs
    sparsity_configs = sparsity_configs_aux
    training_configs = sparsity_configs_aux

    configs_layers_initialization_all_kaiming_relu()
    config_adam_setup()

    initialize_model()
    initialize_training_context()
    initialize_stages_context()
    # CHANGE 
    wandb_initalize(
        Experiment.DEITTINY16IMAGENET,
        type=Tags.TRAIN_PRUNING,
        configs=sparsity_configs,
        other_tags=["ADAM"],
    )
    initialize_dataset_context()
    initalize_training_display()
    MODEL_MODULE.save_weights(f"{PRUNED_MODELS_PATH}/vit_tiny_imagenet_initial_weights.pth")
    acc = 0
    for epoch in range(1, stages_context.args.regrowth_epoch_end + 1):
        epoch_global = epoch
        dataset_context.init_data_split()
       
        train_mixed_pruned_imagenet(
            dataset_context=dataset_context,
            training_context=training_context,
            model=MODEL,
            model_module=MODEL_MODULE,
            training_display=training_display,
        )
        acc = test_pruned_imagenet(
            dataset_context=dataset_context,
            model=MODEL,
            model_module=MODEL_MODULE,
            epoch=get_epoch(),
        )

        stages_context.update_context(
            epoch_global, get_custom_model_sparsity_percent(MODEL_MODULE)
        )
        stages_context.step(training_context)
        MODEL_MODULE.save_weights(f"{PRUNED_MODELS_PATH}/vit_tiny_imagenet_epoch{epoch}_sparsity{get_custom_model_sparsity_percent(MODEL_MODULE):.2f}_acc{acc:.2f}.pth")

    print("Training complete")
    wandb_finish()
