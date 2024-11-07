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


class MaskPruningFunctionSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_param):
        mask = torch.sigmoid(mask_param)
        mask_thresholded = (mask >= 0.5).float()
        ctx.save_for_backward(mask)
        return mask_thresholded

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        # sigmoid derivative
        grad_mask_param = grad_output * mask * (1 - mask)
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
