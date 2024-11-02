Constant Parameters
* `batch_size`: 128
* `learning_rate_weights': 0.0008
* `learning_rate_pruning_mask`: `learning_rate_weights` * 10 
* `learning_rate_flipping_mask`: `learning_rate_weights` * 20

Heuristics:
* `loss_remaining_weights`: scaled by epoch * B

Dynamic Parameters
* B from loss_remaining_weights 
* STOP_EPOCH determining the epoch at which pruning stops