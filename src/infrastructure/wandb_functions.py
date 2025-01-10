import wandb

def wandb_initalize():
    # if WANDB_REGISTER:
    #     wandb.init(
    #         project="resnet50_cifar10",
    #         config={
    #             "batch_size": BATCH_SIZE,
    #             "num_epochs": num_epochs,
    #             "lr_weight_bias": lr_weight_bias,
    #             "lr_custom_params": lr_custom_params,
    #         },
    #     )
    #     wandb.define_metric("epoch")
    #     wandb.define_metric("*", step_metric="epoch")
    pass

def wandb_snapshot():
    #     if WANDB_REGISTER:
    #         wandb.log({"epoch": epoch_global, "test_loss": test_loss, "accuracy": accuracy, "remaining_parameters": remain_percent})
    pass

def wandb_finish():
    wandb.finish()
