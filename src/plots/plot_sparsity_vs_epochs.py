import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define a saturating power-law function with t0
def saturating_power_law(x, L, A, alpha, t0):
    return L + A * (x + t0)**(-alpha)

def load_and_fit(filename):
    with open(filename, 'r') as file:
        values = json.load(file)

    # Normalize your data if needed
    NUMBER_PARAMS = 266100
    values = [(val / NUMBER_PARAMS) * 100 for val in values]
    values = values[10:]

    x_data = np.arange(1, len(values) + 1)  # epochs from 1..N
    y_data = np.array(values, dtype=float)

    # Initial guesses and bounds. Adjust if you see fit problems.
    p0 = [0.1, 10.0, 1.0, 1.0]  # Including t0
    bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])  # Bounds for all params

    try:
        popt, pcov = curve_fit(saturating_power_law, x_data, y_data, p0=p0, bounds=bounds)
        return x_data, y_data, popt
    except RuntimeError as e:
        print(f"Fit did not converge for {filename}: {e}")
        return x_data, y_data, None

def plot_all(files, gammas):
    plt.figure(figsize=(10,6))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'pink', 'brown']

    for i, (f, g) in enumerate(zip(files, gammas)):
        x, y, popt = load_and_fit(f)
        if popt is None:
            continue

        # Generate fine-grained fit curve
        x_fit = np.linspace(1, len(y), 500)
        y_fit = saturating_power_law(x_fit, *popt)

        # Downsample points for clarity
        sample_indices = np.logspace(0, np.log10(len(y)-1), num=200, dtype=int)
        plt.scatter(x[sample_indices], y[sample_indices],
                    color=colors[i], label=f"$\\gamma={g}$", s=50)
        plt.plot(x_fit, y_fit, color=colors[i], linestyle='--', alpha=0.8)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Sparsity Level (%)')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.show()

# Example usage
gammas = [1]
abs_path = r"C:\Users\Statia6\projects\XAI_paper\experiments_outputs"
files = [f"{abs_path}\mnist_lenet300_adam_{gamma}_high_lr.json" for gamma in gammas]
plot_all(files, gammas)
