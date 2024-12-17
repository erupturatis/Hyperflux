import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Data values as normal numbers
values = [
    33396., 32833., 27847., 26482., 23166., 19356., 15615., 13501.,
    10902., 8896., 6964., 5342., 3853., 2408., 1316., 875.,
    614., 413., 164., 178.
]
values = [(val / (35000 * 10)) * 100 for val in values]

# X-axis corresponding to 2^-10 to 2^9
exponents = list(range(-10, 10))
x_values = np.array([2 ** exp for exp in exponents])
y_values = np.array(values)

# Remove zero/negative values for log-log fitting
mask = (x_values > 0) & (y_values > 0)
x_log = np.log(x_values[mask])
y_log = np.log(y_values[mask])

# Perform linear regression on log-log data
slope, intercept, _, _, _ = linregress(x_log, y_log)

# Convert back to power-law parameters
alpha_opt = -slope
c_opt = np.exp(intercept)

# Define the power-law model
def power_law(x, c, alpha):
    return c * x ** (-alpha)

# Plot the data and the fitted power-law curve
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, marker='o', linestyle='-', label='Actual Values')
plt.plot(x_values, power_law(x_values, c_opt, alpha_opt), marker='x', linestyle='--',
         label=f'Fitted Power-Law (c={c_opt:.2f}, alpha={alpha_opt:.2f})')

plt.title('Fixed Power-Law Fit for Values vs Powers of 2')
plt.xscale('log')  # Set x-axis to log scale
plt.yscale('log')  # Set y-axis to log scale
plt.xlabel('$2^x$')
plt.ylabel('Values')
plt.ylim(0, 10)  # Limit y-axis from 0 to 10
plt.grid(True)
plt.legend()
plt.show()

print(f"Optimized parameters: c = {c_opt:.2f}, alpha = {alpha_opt:.2f}")
