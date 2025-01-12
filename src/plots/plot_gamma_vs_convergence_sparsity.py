import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import glob
import os
import torch
from src.infrastructure.constants import SAVED_RESULTS_PATH
from src.infrastructure.others import prefix_path_with_root

def log_model(x, a, b, d):
    lx = np.log(x)
    return a + b * lx + d * (lx ** 2)

def variable_exponent_model(x, c, alpha0, alpha1):
    lx = np.log(x)
    return c * np.exp(- (alpha0 * lx + alpha1 * lx * lx))

saved_results_path = prefix_path_with_root(SAVED_RESULTS_PATH)
exponents = list(range(-6, 9))
gamma_values = []
final_sparsity = []

for exponent in exponents:
    gamma = 2 ** exponent
    gamma_str = "{:.15f}".format(gamma).rstrip('0').rstrip('.') if '.' in "{:.15f}".format(gamma) else "{:.15f}".format(gamma)
    filename = f"mnist_lenet300_adam_{gamma_str}.json"
    file_path = os.path.join(saved_results_path, filename)
    if not os.path.isfile(file_path):
        print(f"Warning: File '{filename}' does not exist in '{saved_results_path}'. Skipping this gamma value.")
        continue
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode JSON from file '{filename}'. Skipping this file.")
            continue
    if isinstance(data, list):
        sparsity_levels = np.array(data)
    elif isinstance(data, dict):
        sparsity_levels = np.array(data.get("sparsity_levels", []))
    else:
        print(f"Warning: Unexpected data format in file '{filename}'. Skipping this file.")
        continue
    if sparsity_levels.size == 0:
        print(f"Warning: No sparsity data found in file '{filename}'. Skipping this file.")
        continue
    final = sparsity_levels[-1]
    gamma_values.append(gamma)
    final_sparsity.append(final)

if not gamma_values:
    raise ValueError("No valid data loaded. Please check the JSON files and their naming convention.")

gamma_values = np.array(gamma_values)
final_sparsity = np.array(final_sparsity)

sorted_indices = np.argsort(gamma_values)
gamma_sorted = gamma_values[sorted_indices]
final_sparsity_sorted = final_sparsity[sorted_indices]

mask = (gamma_sorted > 0) & (final_sparsity_sorted > 0)
gamma_fit = gamma_sorted[mask]
sparsity_fit = final_sparsity_sorted[mask]

if len(gamma_fit) < 3:
    raise ValueError("Not enough data points for curve fitting. Need at least 3.")

p0 = [np.log(sparsity_fit[0]), -1.0, 0.0]

try:
    popt, pcov = curve_fit(log_model, gamma_fit, np.log(sparsity_fit), p0=p0)
    a_fit, b_fit, d_fit = popt
    c_fit = np.exp(a_fit)
    alpha0 = -b_fit
    alpha1 = -d_fit
    print(f"Fitted parameters: c = {c_fit:.4f}, α0 = {alpha0:.4f}, α1 = {alpha1:.4f}")
    gamma_curve = np.logspace(np.log10(gamma_fit.min()), np.log10(gamma_fit.max()), 100)
    alpha0 = 0.68
    alpha1 = 0.0
    sparsity_curve = variable_exponent_model(gamma_curve, c_fit, alpha0, alpha1)
except Exception as e:
    print(f"Curve fitting failed: {e}")
    c_fit, alpha0, alpha1 = 1.0, 1.0, 1.0
    gamma_curve = np.logspace(np.log10(gamma_fit.min()), np.log10(gamma_fit.max()), 100)
    sparsity_curve = variable_exponent_model(gamma_curve, c_fit, alpha0, alpha1)

plt.figure(figsize=(10, 6))
plt.plot(gamma_sorted, final_sparsity_sorted, 's', markersize=5, label='Final Sparsity', color='blue', alpha=0.7)
plt.plot(gamma_curve, sparsity_curve, linestyle='-', linewidth=2, label='Fitted Curve', color='black', alpha=0.7)
plt.xscale('log')
plt.yscale('log')
plt.ylim(final_sparsity_sorted.min() * 0.8, final_sparsity_sorted.max() * 1.2)
plt.xlabel("Gamma (γ)", fontsize=14)
plt.ylabel("Final Sparsity Level (%)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='lower left')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.title("MNIST Pruning Final Sparsity vs. Gamma", fontsize=16)
plt.tight_layout()
plot_save_path = prefix_path_with_root("final_mnist_pruning_plot.pdf")
plt.savefig(plot_save_path, bbox_inches='tight', dpi=300)
plt.show()
print(f"Fitted parameters: c = {c_fit:.4f}, α0 = {alpha0:.4f}, α1 = {alpha1:.4f}")
