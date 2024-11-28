import torch

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
        grad_mask_param = grad_output * mask * (1 - mask)
        return grad_mask_param

# Initialize parameters
mask_param = torch.tensor([-2.0], requires_grad=True)
weight = torch.tensor([3.0], requires_grad=True)
input_val = torch.tensor([4.0], requires_grad=True)

# Forward pass
mask_thresholded = MaskPruningFunctionSigmoid.apply(mask_param)
output = mask_thresholded * weight * input_val  # Output: 0 * 3 * 4 = 0
loss = output.sum()
loss.backward()

# Check gradients
print("mask_param.grad:", mask_param.grad)  # Expected: non-zero
print("weight.grad:", weight.grad)          # Expected: 0 (since output doesn't depend on weight when mask is 0)
print("input_val.grad:", input_val.grad)    # Expected: 0 (similarly)
