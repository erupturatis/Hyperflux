import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json

# Load data from JSON file
with open('results_mnist_pruning_vs_epochs.json', 'r') as file:
    values = json.load(file)

# Normalize values
values = [(val / values[0]) * 100 for val in values]
values = values[:900]

# Prepare data for fitting
epochs = np.arange(len(values))
x_values = epochs + 1  # Shift by 1 to avoid zero in log
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

# Sample relevant data points
sample_indices = np.logspace(0, np.log10(len(y_values) - 1), num=20, dtype=int)
sample_x = x_values[sample_indices]
sample_y = np.array(y_values)[sample_indices]

# Plot the fit and sampled points
plt.figure(figsize=(10, 6))
plt.plot(x_fit, y_fit, color='blue', linestyle='--', linewidth=2, alpha=0.8)
plt.scatter(sample_x, sample_y, color='blue', s=50, label="Lenet-300, $\gamma=2.0$", alpha=0.9)

# Apply logarithmic scales
plt.xscale('log')
plt.yscale('log')

# Set limits
plt.xlim(1, 1000)
plt.ylim(1e-2, 1e2)

# Custom ticks for both axes
x_ticks = [1, 10, 50, 100, 200, 500, 1000]
x_labels = [f"{xtick}" for xtick in x_ticks]
plt.xticks(x_ticks, x_labels, fontsize=12)

y_ticks = [0.01, 0.1, 1, 10, 100]
y_labels = [f"{ytick}" for ytick in y_ticks]
plt.yticks(y_ticks, y_labels, fontsize=12)

# Customize labels
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Sparsity Level %", fontsize=14)

# Add legend
plt.legend(fontsize=12, loc='lower left')

# Add grid and adjust layout
plt.grid(True, linestyle='--', linewidth=0.5)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# Save and show plot
plt.savefig("log_log_sampled_fit_plot.pdf", bbox_inches='tight', dpi=300)
plt.show()

# Print fitted parameters
print("Fitted parameters:")
print(f"c     = {c_fit:.4g}")
print(f"alpha = {alpha_fit:.4g}")
print(f"beta  = {beta_fit:.4g}")
print(f"k     = {k_fit:.4g}")
print(f"L     = {L_fit:.4g}")
