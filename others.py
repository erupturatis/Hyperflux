import torch
import matplotlib.pyplot as plt
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Simulate learning rate behavior
model_params = [torch.nn.Parameter(torch.randn(2, 2))]
optimizer = SGD(model_params, lr=0.1)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1.25)

# Store learning rate for 200 epochs
lr_values = []
epochs = 200

for epoch in range(epochs):
    scheduler.step()
    lr_values.append(optimizer.param_groups[0]['lr'])

# Plot learning rate
plt.plot(range(epochs), lr_values)
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Cosine Annealing with Warm Restarts (T_0=50)")
plt.grid()
plt.show()
