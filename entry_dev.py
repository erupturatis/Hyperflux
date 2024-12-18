import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Data values as normal numbers
# decay LR 0.95, 0.0005 LR masks, 300 epochs
# values = [62401.0, 52114.0, 47292.0, 38099.0, 30758.0, 25624.0, 17366.0, 13202.0, 9717.0, 7604.0, 4886.0, 3142.0, 2221.0, 1264.0, 738.0, 476.0, 313.0, 217.0, 129.0]
# values = [62401.0, 52114.0, 47292.0, 38099.0, 30758.0, 25624.0, 17366.0, 13202.0, 9717.0, 7604.0, 4886.0, 3142.0, 2221.0, 1264.0, 738.0, 476.0, 313.0, 217.0, 129.0, 29.0, 0.0]

# decay LR 0.99, 0.00075 LR masks, 400 epochs, -15, 11
# values = [150544.0, 72768.0, 67566.0, 67062.0, 65995.0, 59276.0, 47173.0, 42069.0, 39783.0, 32073.0, 25624.0, 18919.0, 13565.0, 10027.0, 7277.0, 5248.0, 3249.0, 2227.0, 1314.0, 743.0, 470.0, 325.0, 211.0, 123.0, 45.0, 13.0]

# decay LR 0.99, 0.0005 LR masks, 400 epochs, -15, 11
values = [222651.0, 92206.0, 77633.0, 76593.0, 75170.0, 68716.0, 60095.0, 55670.0, 48943.0, 39427.0, 28878.0, 22694.0, 15290.0, 12001.0, 7830.0, 5604.0, 3738.0, 2450.0, 1429.0, 801.0, 517.0, 335.0, 205.0, 154.0, 85.0, 11.0]

values = [(val / 266610) * 100 for val in values]

# X-axis corresponding to 2^-10 to 2^9
exponents = list(range(-15, 11))
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
# alpha_opt = 0.5436
c_opt = np.exp(intercept)

# Define the power-law model
def power_law(x, c, alpha):
    return c * x ** (-alpha)

# Plot the data and the fitted power-law curve
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, marker='o', linestyle='-', label='Actual Values')
plt.plot(x_values, power_law(x_values, c_opt, alpha_opt), marker='x', linestyle='--',
         label=f'Fitted Power-Law (c={c_opt:.4f}, alpha={alpha_opt:.4f})')

plt.title('Fixed Power-Law Fit for Values vs Powers of 2')
plt.xscale('log')  # Set x-axis to log scale
plt.yscale('log')  # Set y-axis to log scale
plt.xlabel('$2^x$')
plt.ylabel('Values')
plt.ylim(0, 100)  # Limit y-axis from 0 to 10
plt.grid(True)
plt.legend()
plt.show()

print(f"Optimized parameters: c = {c_opt:.2f}, alpha = {alpha_opt:.2f}")
