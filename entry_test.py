import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json

values = []

with open('experiments_outputs/results_mnist_pruning_vs_epochs_g0.5.json', 'r') as file:
    values = json.load(file)

values = [(val / values[0])*100 for val in values]
values = values[:400]

epochs = np.arange(len(values))
x_values = epochs + 1  # shift by 1 to avoid zero in log
y_values = values

# Define the model: Power Law + Exponential + Offset
def power_law_exp_with_offset(x, c, alpha, beta, k, L):
    return c * x**(-alpha) + beta * np.exp(-k * x) + L

# Initial guess for parameters [c, alpha, beta, k, L]
p0 = [1, 1, -2, 0.1, 1.2]

# Fit the model
popt, pcov = curve_fit(
    power_law_exp_with_offset, x_values, y_values, p0=p0, maxfev=10000
)
c_fit, alpha_fit, beta_fit, k_fit, L_fit = popt

# Generate curve for plotting
x_fit = np.linspace(1, len(y_values), 1000)
y_fit = power_law_exp_with_offset(x_fit, c_fit, alpha_fit, beta_fit, k_fit, L_fit)

# Identify convergence point (example: last lowest in data)
last_lowest_x = x_values[-1]
last_lowest_y = min(y_values)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, 'b.', label='Data')
plt.plot(x_fit, y_fit, 'r-', label='Fit')
plt.scatter(last_lowest_x, last_lowest_y, color='g', zorder=5, label='Convergence')
plt.axvline(last_lowest_x, color='g', ls='--')
plt.axhline(last_lowest_y, color='g', ls='--')
plt.xscale('log')
plt.yscale('log')
plt.title('Power Law + Exponential + Offset Fit')
plt.xlabel('Epochs')
plt.ylabel('Values')
plt.legend()
plt.show()

print("Fitted parameters:")
print(f"c     = {c_fit:.4g}")
print(f"alpha = {alpha_fit:.4g}")
print(f"beta  = {beta_fit:.4g}")
print(f"k     = {k_fit:.4g}")
print(f"L     = {L_fit:.4g}")
