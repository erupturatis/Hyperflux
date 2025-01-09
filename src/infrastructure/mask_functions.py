import numpy as np
import torch

class MaskFlipFunctionTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_param):
        mask = torch.tanh(mask_param)
        ctx.save_for_backward(mask)
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        # tanh derivative
        grad_mask_param = grad_output * (1 - mask**2)
        return grad_mask_param

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


class MaskPruningFunctionSigmoidDebugging(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_param, weights):
        mask = torch.sigmoid(mask_param)
        mask_prob = custom_sigmoid(mask_param, 20)
        mask_thresholded = (mask >= 0.5).float()

        print("Mask values greater than 0 after thresholding:", mask[mask >= 0.5][:10])
        # print("Mask values greater than 0 after thresholding:", mask_prob[mask >= 0.5][:10])
        print("Weights values greater than 0 after thresholding:", weights[mask >= 0.5])

        ctx.save_for_backward(mask)
        return mask_thresholded

    @staticmethod
    def backward(ctx, grad_output):
        pass

INFERENCE = {
    "inference": False
}

idx = {
    "cnt": 1
}


# p = 0.25
# p = 0
# if not INFERENCE["inference"]:
#     # dropout_mask = torch.bernoulli(torch.full_like(mask_thresholded, 1-p))
#     dropout_mask = torch.bernoulli(1-mask_prob/100)
#     mask_thresholded *= dropout_mask

# if not INFERENCE["inference"]:
#     # Apply conditional modification
#     modification_mask = torch.bernoulli(1-mask_prob/10)
#     # modification_mask = torch.bernoulli(torch.full_like(mask_thresholded, mask_prob))
#     mask_thresholded = mask_thresholded * (1 - modification_mask) + mask_thresholded * mask * modification_mask

# class MaskPruningFunctionSigmoidVanilla(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, mask_param):
#         # mask = torch.sigmoid(mask_param)
#         mask = mask_param
#         mask_thresholded = (mask >= 0).float()
#         # mask_thresholded = (mask >= 0.5).float()
#         ctx.save_for_backward(mask)
#         return mask_thresholded
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         mask, = ctx.saved_tensors
#         grad_mask_param = grad_output * mask * (1 - mask)
#         return grad_mask_param

class MaskPruningFunctionSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_param):
        mask = mask_param
        mask_thresholded = (mask >= 0).float()

        ctx.save_for_backward(mask, mask_thresholded)
        return mask_thresholded

    @staticmethod
    def backward(ctx, grad_output):
        mask, _ = ctx.saved_tensors

        mask = torch.sigmoid(mask)
        # grad_mask_param = grad_output * 1
        grad_mask_param = grad_output * mask * (1 - mask)

        return grad_mask_param


class MaskPruningFunctionSigmoidTest(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_param):
        mask = mask_param
        mask_thresholded = (mask >= 0).float()

        ctx.save_for_backward(mask, mask_thresholded)
        return mask_thresholded

    @staticmethod
    def backward(ctx, grad_output):
        mask, mask_thresholded = ctx.saved_tensors

        non_masked = mask_thresholded
        masked = 1 - mask_thresholded

        mask = torch.sigmoid(mask)

        abs_grad = -torch.abs(grad_output * mask * (1 - mask)) * non_masked
        regrowing_grad = grad_output * mask * (1 - mask) * masked

        grad_mask_param = abs_grad + regrowing_grad
        # grad_mask_param = grad_output * mask * (1 - mask)

        return grad_mask_param


datablob = {
    "grad_unpruned": [],
    "registered_indices": []
}

def custom_sigmoid(x, k=1.0):
    return 1 / (1 + torch.exp(-k * x))

class MaskPruningFunctionSigmoidSampled(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_param):
        # mask_prob = custom_sigmoid(mask_param, 1)
        # sampled_mask = torch.bernoulli(mask_prob)

        mask = torch.sigmoid(mask_param)
        mask_thresholded = (mask >= 0.5).float()
        # Save mask probabilities for backward pass
        ctx.save_for_backward(mask)
        return mask_thresholded

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
