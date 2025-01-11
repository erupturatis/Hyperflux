import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Provided test_sparsity data
test_sparsity = [
    19.85, 3.71, 2.71, 2.28, 2.03, 1.85, 1.72, 1.63, 1.59, 1.53,
    1.49, 1.45, 1.42, 1.41, 1.38, 1.36, 1.35, 1.34, 1.32, 1.31,
    1.30, 1.29, 1.29, 1.29, 1.27, 1.27, 1.26, 1.26, 1.25, 1.25,
    1.25, 1.24, 1.24, 1.24, 1.24, 1.24, 1.24, 1.23, 1.23, 1.23,
    1.23, 1.23, 1.23, 1.22, 1.22, 1.22, 1.22, 1.22, 1.21, 1.22,
    1.21, 1.22, 1.21, 1.22, 1.21, 1.21, 1.21, 1.21, 1.21, 1.21,
    1.21, 1.21, 1.20, 1.20, 1.21, 1.20, 1.20, 1.20
]

# Generate epoch numbers from 1 to 68
epochs = np.arange(1, len(test_sparsity) + 1)
sparsity = np.array(test_sparsity)

# Filter out any non-positive values to avoid log issues (if any)
positive_indices = sparsity > 0
epochs = epochs[positive_indices]
sparsity = sparsity[positive_indices]

# Log-transform the data
log_epochs = np.log(epochs)
log_sparsity = np.log(sparsity)

# Prepare the design matrix for quadratic regression
X = np.vstack((log_epochs, log_epochs**2)).T
Y = log_sparsity

# Perform quadratic regression using numpy's polyfit (degree=2)
coeffs = np.polyfit(log_epochs, log_sparsity, deg=2)
b2, b1, b0 = coeffs  # numpy returns highest degree first

# Define the quadratic function based on the fit
def quadratic_fit(x, b0, b1, b2):
    return b0 + b1 * x + b2 * x**2

# Generate fitted log(sparsity) values
fitted_log_sparsity = quadratic_fit(log_epochs, b0, b1, b2)

# Convert back to original scale
fitted_sparsity = np.exp(fitted_log_sparsity)

# Calculate R-squared
ss_res = np.sum((Y - fitted_log_sparsity) ** 2)
ss_tot = np.sum((Y - np.mean(Y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Create the log-log plot
plt.figure(figsize=(10, 6))
plt.loglog(epochs, sparsity, marker='o', linestyle='-', color='b', label='Data')
plt.loglog(epochs, fitted_sparsity, linestyle='--', color='r', label='Quadratic Fit')

# Add titles and labels
plt.title('Test Sparsity over Epochs with Quadratic Log-Log Fit')
plt.xlabel('Epoch')
plt.ylabel('Sparsity')

# Add grid for better readability
plt.grid(True, which="both", ls="--", linewidth=0.5)

# Add legend to differentiate data and fit
plt.legend()

# Display R-squared on the plot for reference
plt.text(0.05, 0.95, f'$R^2$ = {r_squared:.4f}', transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()

# Extract parameters
ln_c = b0
alpha0 = -b1
alpha1 = -b2

c = np.exp(ln_c)

# Print the fitted parameters
print(f"Fitted Quadratic Log-Log Model:")
print(f"ln(sparsity) = ln(c) - alpha0 * ln(epoch) - alpha1 * (ln(epoch))^2")
print(f"ln(c) = {ln_c:.4f}")
print(f"alpha0 = {alpha0:.4f}")
print(f"alpha1 = {alpha1:.4f}")
print(f"Coefficient of Determination (R^2): {r_squared:.4f}")
