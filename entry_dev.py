import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid-based function e(c) with parameters 'k' and 'c0'
def e(c, param, k, c0):
    sigmoid = 1 / (1 + np.exp(-k * (c - c0)))
    return param + (0.999 - param) * sigmoid

# Define the function f(a) using logarithms for numerical stability
def f(a, param, k, c0):
    # Compute log(e(k)) for k from 1 to a
    log_e_values = [np.log(e(epoch, param, k, c0)) for epoch in range(1, a + 1)]
    # Sum the logs
    log_product = np.sum(log_e_values)
    # Compute log(f(a)) = log(100) + log_product
    log_fa = np.log(100) + log_product
    # Exponentiate to get f(a)
    return np.exp(log_fa)

# Define the classic exponential function f_exp(a)
def f_exp(a, lam):
    return 100 * np.exp(-lam * a)

# Binary search to find the optimal 'param' for custom function
def binary_search_param(target, k, c0, a, tol=1e-6, max_iter=100):
    lower = 0.0
    upper = 1.0
    iteration = 0
    best_param = None
    best_fa = None

    while iteration < max_iter:
        mid = (lower + upper) / 2
        current_fa = f(a, mid, k, c0)

        # Debug: Print current state
        print(f"Iteration {iteration+1}: param={mid:.6f}, f({a})={current_fa:.6f}")

        # Check if the current f(a) is close enough to the target
        if abs(current_fa - target) < tol:
            best_param = mid
            best_fa = current_fa
            break

        # Decide which half to choose for the next iteration
        if current_fa < target:
            # Assuming f(a) increases with param
            lower = mid
        else:
            upper = mid

        # Update the best found so far
        if best_fa is None or abs(current_fa - target) < abs(best_fa - target):
            best_param = mid
            best_fa = current_fa

        iteration += 1

    return best_param, best_fa

# Function to calculate lambda for classic exponential
def calculate_lambda(target, epochs_num):
    # Ensure target is positive and less than or equal to 100
    if target <= 0 or target > 100:
        raise ValueError("Target must be between 0 (exclusive) and 100 (inclusive).")
    lam = -np.log(target / 100) / epochs_num
    return lam

# Function to compute f_exp(a) using the calculated lambda
def compute_f_exp(a_values, lam):
    return 100 * np.exp(-lam * a_values)

# Function to compute f(a) for plotting with given param, k, and c0
def compute_f_custom(a_values, param, k, c0):
    return [f(a, param, k, c0) for a in a_values]

# Example usage
if __name__ == "__main__":
    # Define your target for f(200)
    target_final = 0.5  # Replace with your desired target between 1 and 100

    # Define parameters for the sigmoid function
    k = 0.05    # Controls the steepness of the transition
    c0 = 100    # Midpoint of the transition

    epochs_num = 400

    # Perform binary search to find the optimal 'param' for custom function
    print(f"Performing binary search with k={k}, c0={c0} to reach target f(200)={target_final}\n")
    optimal_param, achieved_f200 = binary_search_param(target=target_final, k=k, c0=c0, a=epochs_num, tol=1e-6, max_iter=100)

    print("\nOptimal Parameter Found for Custom Function:")
    print(f"param = {optimal_param:.6f}")
    print(f"f(200) = {achieved_f200:.6f} (Target: {target_final})\n")

    # Calculate lambda for classic exponential
    lam = calculate_lambda(target_final)
    print("Parameter for Classic Exponential Function:")
    print(f"lambda = {lam:.6f}\n")

    # Generate values for a from 1 to 200
    a_values = np.arange(1, epochs_num + 1)
    f_custom_values = compute_f_custom(a_values, optimal_param, k, c0)
    f_exp_values = compute_f_exp(a_values, lam)

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(a_values, f_custom_values, marker='o', linestyle='-', color='b',
             label=f'Custom Function (param = {optimal_param:.4f})')
    plt.plot(a_values, f_exp_values, marker='x', linestyle='--', color='g',
             label=f'Classic Exponential (λ = {lam:.4f})')
    plt.axhline(y=target_final, color='r', linestyle='--',
                label=f'Target f(200) = {target_final}')
    plt.title('Comparison of Modified Custom Function and Classic Exponential Function')
    plt.xlabel('Epoch (a)')
    plt.ylabel('f(a)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
