import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import glob
import os
import torch
from itertools import cycle
from src.infrastructure.constants import SAVED_RESULTS_PATH
from src.infrastructure.others import prefix_path_with_root

def log_model(x, a, b, d):
    lx = np.log(x)
    return a + b * lx + d * (lx ** 2)

def variable_exponent_model(x, c, alpha0, alpha1):
    lx = np.log(x)
    return c * np.exp(- (alpha0 * lx + alpha1 * lx * lx))

def format_gamma(gamma):
    """Format gamma to remove trailing zeros and decimal point if unnecessary."""
    gamma_str = "{:.15f}".format(gamma).rstrip('0').rstrip('.') if '.' in "{:.15f}".format(gamma) else "{:.15f}".format(gamma)
    return gamma_str

def process_dataset(base_filename_pattern, exponents, saved_results_path):
    gamma_values = []
    final_sparsity = []
    for exponent in exponents:
        gamma = 2 ** exponent
        gamma_str = format_gamma(gamma)
        filename = base_filename_pattern.format(gamma=gamma_str)
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
        print(f"Error: No valid data loaded for pattern '{base_filename_pattern}'. Skipping this dataset.")
        return None, None, None, None

    gamma_values = np.array(gamma_values)
    final_sparsity = np.array(final_sparsity)

    sorted_indices = np.argsort(gamma_values)
    gamma_sorted = gamma_values[sorted_indices]
    final_sparsity_sorted = final_sparsity[sorted_indices]

    mask = (gamma_sorted > 0) & (final_sparsity_sorted > 0)
    gamma_fit = gamma_sorted[mask]
    sparsity_fit = final_sparsity_sorted[mask]

    if len(gamma_fit) < 3:
        print(f"Warning: Not enough data points for curve fitting in pattern '{base_filename_pattern}'. Skipping curve fitting.")
        return gamma_sorted, final_sparsity_sorted, None, None

    p0 = [np.log(sparsity_fit[0]), -1.0, 0.0]

    try:
        popt, pcov = curve_fit(log_model, gamma_fit, np.log(sparsity_fit), p0=p0)
        a_fit, b_fit, d_fit = popt
        c_fit = np.exp(a_fit)
        alpha0 = -b_fit
        alpha1 = -d_fit

        alpha0 = 0.68
        alpha1 = 0
        print(f"Fitted parameters for '{base_filename_pattern}': c = {c_fit:.4f}, α0 = {alpha0:.4f}, α1 = {alpha1:.4f}")
        gamma_curve = np.logspace(np.log10(gamma_fit.min()), np.log10(gamma_fit.max()), 100)
        sparsity_curve = variable_exponent_model(gamma_curve, c_fit, alpha0, alpha1)
        return gamma_sorted, final_sparsity_sorted, gamma_curve, sparsity_curve
    except Exception as e:
        print(f"Curve fitting failed for pattern '{base_filename_pattern}': {e}")
        return gamma_sorted, final_sparsity_sorted, None, None

def main():
    saved_results_path = prefix_path_with_root(SAVED_RESULTS_PATH)

    # Define the range of exponents from -9 to 9 inclusive
    exponents = list(range(-9, 10))

    # Define multiple base filename patterns with a placeholder for gamma
    # Modify this list based on your actual filename patterns
    base_filename_patterns = [
        "mnist_lenet300_adam_{gamma}.json",
        "mnist_lenet300_sgd_{gamma}.json",
        "cifar10_resnet18_adam_{gamma}.json",
        "cifar10_resnet18_sgd_{gamma}.json",
        # Add more patterns as needed, e.g.,
        # "cifar_lenet_adam_{gamma}.json",
        # "imagenet_resnet_adam_{gamma}.json",
    ]

    # Initialize the plot
    plt.figure(figsize=(12, 8))

    # Create a color cycle iterator
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    for base_pattern in base_filename_patterns:
        color = next(color_cycle)
        gamma_sorted, final_sparsity_sorted, gamma_curve, sparsity_curve = process_dataset(base_pattern, exponents, saved_results_path)
        if gamma_sorted is None or final_sparsity_sorted is None:
            continue

        label_data = f"{base_pattern.replace('{gamma}.json', '')} Data"
        plt.scatter(gamma_sorted, final_sparsity_sorted, s=50, label=label_data, color=color, alpha=0.7, marker='s')

        if gamma_curve is not None and sparsity_curve is not None:
            label_curve = f"{base_pattern.replace('{gamma}.json', '')} Fit"
            plt.plot(gamma_curve, sparsity_curve, linestyle='-', linewidth=2, label=label_curve, color=color, alpha=0.7)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Gamma (γ)", fontsize=14)
    plt.ylabel("Final Sparsity Level (%)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc='lower left')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.title("Pruning Final Sparsity vs. Gamma", fontsize=16)
    plt.tight_layout()
    plot_save_path = prefix_path_with_root("final_pruning_plot.pdf")
    plt.savefig(plot_save_path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Plot saved to '{plot_save_path}'.")

if __name__ == "__main__":
    main()
