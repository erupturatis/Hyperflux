import torch


class MaskFlipFunctionSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_param):
        mask = torch.sigmoid(mask_param)
        mask_classified = torch.where(mask < 0.5, -1,1)
        ctx.save_for_backward(mask)
        return mask_classified.float()

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        # sigmoid derivative
        grad_mask_param = grad_output * mask * (1 - mask)
        return grad_mask_param



def set_inference(val:bool):
    global INFERENCE
    INFERENCE["inference"] = val

INFERENCE = {
    "inference": False,
}

class MaskPruningFunctionSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_param):
        mask = torch.sigmoid(mask_param)
        mask_thresholded = (mask >= 0.5).float()

        inference = INFERENCE["inference"]
        p = 0.02
        if not inference:
            dropout_mask = torch.bernoulli(torch.full_like(mask_thresholded, 1-p))
            mask_thresholded *= dropout_mask

        ctx.save_for_backward(mask)
        return mask_thresholded

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        # sigmoid derivative
        grad_mask_param = grad_output * mask * (1 - mask)
        return grad_mask_param

class MaskPruningFunctionSigmoidSampled(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_param):
        mask_prob = torch.sigmoid(mask_param)
        sampled_mask = torch.bernoulli(mask_prob)
        # Save mask probabilities for backward pass
        ctx.save_for_backward(mask_prob)
        # Straight-Through Estimator:
        # Forward pass uses sampled_mask
        # Backward pass uses mask_prob gradients
        return sampled_mask.float()

    @staticmethod
    def backward(ctx, grad_output):
        mask_prob, = ctx.saved_tensors
        grad_mask_param = grad_output * mask_prob * (1 - mask_prob)

        return grad_mask_param


class MaskFlipFunctionLeaky(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_param):
        # Apply Leaky ReLU activation to mask_param
        mask = torch.nn.functional.leaky_relu(mask_param, negative_slope=0.01)
        mask_classified = torch.where(mask < 0, -1, 1)
        ctx.save_for_backward(mask_param)
        return mask_classified.float()

    @staticmethod
    def backward(ctx, grad_output):
        mask_param, = ctx.saved_tensors
        grad_mask_param = grad_output * ((mask_param > 0).float() + 0.01 * (mask_param <= 0).float())
        return grad_mask_param


class MaskPruningFunctionLeaky(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_param):
        mask = torch.nn.functional.leaky_relu(mask_param, negative_slope=0.01)
        mask_thresholded = (mask >= 0).float()
        ctx.save_for_backward(mask_param)
        return mask_thresholded

    @staticmethod
    def backward(ctx, grad_output):
        mask_param, = ctx.saved_tensors
        grad_mask_param = grad_output * ((mask_param > 0).float() + 0.01 * (mask_param <= 0).float())
        return grad_mask_param

class MaskPruningFunctionLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_param):
        mask = (mask_param > 0).float()
        ctx.save_for_backward(mask_param)
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: gradient is passed through as is
        grad_mask_param = grad_output.clone()
        return grad_mask_param
