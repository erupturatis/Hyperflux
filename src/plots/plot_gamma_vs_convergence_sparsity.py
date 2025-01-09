bimport numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ===========================
# Data Preparation for ResNet18
# ===========================
values_resnet18_first_half = [
    10064085.0, 10055008.0, 10034062.0, 9996538.0, 9893478.0,
    9464791.0, 7728262.0, 6821013.0, 4729235.0, 3722127.0,
    2891457.0, 2164455.0
]
exponents_resnet18_first_half = [2**-15, 2**-14, 2**-13, 2**-12, 2**-11, 2**-10,
                                 2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4]
values_resnet18_second_half = [
    1142904.0, 806090.0, 567648.0, 400349.0, 280996.0,
    193526.0, 127653.0, 75962.0, 41294.0, 21116.0,
    10434.0, 5024.0, 2444.0
]
exponents_resnet18_second_half = [2**-2, 2**-1, 2**0, 2**1, 2**2,
                                  2**3, 2**4, 2**5, 2**6, 2**7,
                                  2**8, 2**9, 2**10]

# Combine first and second half data
values_resnet18 = values_resnet18_first_half + values_resnet18_second_half
exponents_resnet18 = exponents_resnet18_first_half + exponents_resnet18_second_half

# Normalize values to percentage
values_resnet18 = [(val / 11000000) * 100 for val in values_resnet18]

# Convert to NumPy arrays for convenience
x_resnet18 = np.array(exponents_resnet18)
y_resnet18 = np.array(values_resnet18)

# Filter out non-positive values for log scale
mask_resnet18 = (x_resnet18 > 0) & (y_resnet18 > 0)
x_data_resnet18 = x_resnet18[mask_resnet18]
y_data_resnet18 = y_resnet18[mask_resnet18]

# ===========================
# Curve Fitting for ResNet18
# ===========================
def log_model(x, a, b, d):
    lx = np.log(x)
    return a + b * lx + d * (lx ** 2)

# Initial guess for the parameters
p0_resnet18 = [np.log(y_data_resnet18[0]), -1.0, 0.0]

# Perform curve fitting
popt_resnet18, _ = curve_fit(log_model, x_data_resnet18, np.log(y_data_resnet18), p0=p0_resnet18)
a_fit_resnet18, b_fit_resnet18, d_fit_resnet18 = popt_resnet18

# Extract fitted parameters
c_fit_resnet18 = np.exp(a_fit_resnet18)
alpha0_resnet18 = -b_fit_resnet18
alpha1_resnet18 = -d_fit_resnet18

def variable_exponent_model(x, c, alpha0, alpha1):
    lx = np.log(x)
    return c * np.exp(- (alpha0 * lx + alpha1 * lx * lx))

# ===========================
# Generate Synthetic MNIST Data
# ===========================
# Set c = 1.2 for MNIST
c_mnist = 1.2

# Compute the MNIST sparsity levels using the ResNet18 fitted alphas
y_mnist_clean = variable_exponent_model(x_resnet18, c_mnist, alpha0_resnet18, alpha1_resnet18)

# Introduce noise similar to ResNet18 data (1% noise)
np.random.seed(42)  # For reproducibility
noise_mnist = 0.01 * y_mnist_clean * np.random.randn(len(y_mnist_clean))
y_mnist_noisy = y_mnist_clean + noise_mnist

# Ensure no negative values after adding noise
y_mnist_noisy = np.clip(y_mnist_noisy, a_min=1e-3, a_max=None)

# ===========================
# Plotting Both Datasets
# ===========================
plt.figure(figsize=(10, 6))  # Figure size

# Plot ResNet18 actual data with small blue bullet points
plt.plot(x_resnet18, y_resnet18, 'o', markersize=5, label='ResNet18 Actual Data', color='blue', alpha=0.7)

# Plot ResNet18 approximation curve as a solid blue line
y_resnet18_fit = variable_exponent_model(x_resnet18, c_fit_resnet18, alpha0_resnet18, alpha1_resnet18)
plt.plot(x_resnet18, y_resnet18_fit, linestyle='-', linewidth=2, label='ResNet18 Approximation', color='blue', alpha=0.7)

# Plot MNIST synthetic data with small red bullet points
plt.plot(x_resnet18, y_mnist_noisy, 's', markersize=5, label='MNIST Synthetic Data', color='red', alpha=0.7)

# Plot MNIST approximation curve as a dashed red line
y_mnist_fit = variable_exponent_model(x_resnet18, c_mnist, alpha0_resnet18, alpha1_resnet18)
plt.plot(x_resnet18, y_mnist_fit, linestyle='--', linewidth=2, label='MNIST Approximation', color='red', alpha=0.7)

# Customize plot
plt.xscale('log')        # Logarithmic x-axis
plt.yscale('log')        # Logarithmic y-axis
plt.ylim(1e-2, 1e2)       # Y-axis limits
plt.xlabel("gamma", fontsize=14)             # X-axis label
plt.ylabel("Sparsity Level %", fontsize=14)   # Y-axis label
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='lower left')     # Legend styling
plt.grid(True, linestyle='--', linewidth=0.5)  # Grid styling
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Adjust layout

# Save the plot as a PDF
plt.savefig("combined_plot.pdf", bbox_inches='tight', dpi=300)
plt.show()

# ===========================
# Print Fitted Parameters
# ===========================
print(f"ResNet18 Fitted parameters: c = {c_fit_resnet18:.4f}, α0 = {alpha0_resnet18:.4f}, α1 = {alpha1_resnet18:.4f}")
print(f"MNIST Synthetic parameters: c = {c_mnist}, α0 = {alpha0_resnet18:.4f}, α1 = {alpha1_resnet18:.4f}")
