import numpy as np
import matplotlib.pyplot as plt

# ===========================
# Function Definitions
# ===========================

def sigmoid(c, k, c0):
    """
    Computes the sigmoid function.

    Parameters:
        c (int): Current epoch.
        k (float): Steepness of the sigmoid.
        c0 (float): Midpoint of the sigmoid.

    Returns:
        float: Sigmoid value at epoch c.
    """
    return 1 / (1 + np.exp(-k * (c - c0)))

def e(c, param, k, c0):
    """
    Computes the e(c) function with a sigmoid-based transition.

    Parameters:
        c (int): Current epoch.
        param (float): Baseline parameter (0 <= param <= 1).
        k (float): Steepness of the sigmoid.
        c0 (float): Midpoint of the sigmoid.

    Returns:
        float: e(c) value at epoch c.
    """
    sig = sigmoid(c, k, c0)
    return param * (1 - sig) + 0.999 * sig

def f(a, param, k, c0):
    """
    Computes the f(a) function as the product of e(k) from 1 to a.

    Parameters:
        a (int): Current epoch.
        param (float): Baseline parameter.
        k (float): Steepness of the sigmoid.
        c0 (float): Midpoint of the sigmoid.

    Returns:
        float: f(a) value at epoch a.
    """
    # To prevent numerical underflow, compute in log-space
    log_e_values = [np.log(e(c, param, k, c0)) for c in range(1, a + 1)]
    log_product = np.sum(log_e_values)
    log_fa = np.log(100) + log_product
    return np.exp(log_fa)

def f_exp(a, lam):
    """
    Computes the classic exponential decay function.

    Parameters:
        a (int or array-like): Epoch(s).
        lam (float): Decay rate.

    Returns:
        float or array-like: f_exp(a) value(s).
    """
    return 100 * np.exp(-lam * a)

# ===========================
# Binary Search Implementation
# ===========================

def binary_search_param(target, k, c0, a=200, tol=1e-6, max_iter=100):
    """
    Performs a binary search to find the optimal 'param' that makes f(a) close to target.

    Parameters:
        target (float): Desired target value for f(a) at epoch a.
        k (float): Steepness of the sigmoid.
        c0 (float): Midpoint of the sigmoid.
        a (int): Epoch number to target (default=200).
        tol (float): Tolerance for convergence (default=1e-6).
        max_iter (int): Maximum number of iterations (default=100).

    Returns:
        tuple: (optimal_param, achieved_fa)
    """
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

# ===========================
# Lambda Calculation for Exponential
# ===========================

def calculate_lambda(target):
    """
    Calculates the lambda parameter for the classic exponential function to meet the target at epoch 200.

    Parameters:
        target (float): Desired target value for f(a) at epoch a=200.

    Returns:
        float: Calculated lambda.
    """
    if target <= 0 or target > 100:
        raise ValueError("Target must be between 0 (exclusive) and 100 (inclusive).")
    lam = -np.log(target / 100) / 200
    return lam

# ===========================
# Plotting Functions
# ===========================

def compute_f_custom(a_values, param, k, c0):
    """
    Computes f(a) for a range of epochs.

    Parameters:
        a_values (array-like): Array of epoch numbers.
        param (float): Baseline parameter.
        k (float): Steepness of the sigmoid.
        c0 (float): Midpoint of the sigmoid.

    Returns:
        list: List of f(a) values.
    """
    return [f(a, param, k, c0) for a in a_values]

def compute_f_exp_values(a_values, lam):
    """
    Computes f_exp(a) for a range of epochs.

    Parameters:
        a_values (array-like): Array of epoch numbers.
        lam (float): Decay rate.

    Returns:
        array-like: Array of f_exp(a) values.
    """
    return f_exp(a_values, lam)

# ===========================
# Main Execution
# ===========================

def main():
    # ===========================
    # User Inputs
    # ===========================

    # Define your target for f(200)
    target_f200 = 5  # Replace with your desired target between 1 and 100

    # Define parameters for the sigmoid function
    k = 0.1   # Controls the steepness of the transition
    c0 = 50    # Midpoint of the transition

    # ===========================
    # Binary Search to Find Optimal 'param'
    # ===========================

    print(f"Performing binary search with k={k}, c0={c0} to reach target f(200)={target_f200}\n")
    optimal_param, achieved_f200 = binary_search_param(target=target_f200, k=k, c0=c0, a=200, tol=1e-6, max_iter=100)

    print("\nOptimal Parameter Found for Custom Function:")
    print(f"param = {optimal_param:.6f}")
    print(f"f(200) = {achieved_f200:.6f} (Target: {target_f200})\n")

    # ===========================
    # Calculate Lambda for Classic Exponential
    # ===========================

    lam = calculate_lambda(target_f200)
    print("Parameter for Classic Exponential Function:")
    print(f"lambda = {lam:.6f}\n")

    # ===========================
    # Generate Epoch Values
    # ===========================

    a_values = np.arange(1, 201)  # Epochs from 1 to 200

    # ===========================
    # Compute f(a) Values
    # ===========================

    f_custom_values = compute_f_custom(a_values, optimal_param, k, c0)
    f_exp_values = compute_f_exp_values(a_values, lam)

    # ===========================
    # Plotting
    # ===========================

    plt.figure(figsize=(12, 7))
    plt.plot(a_values, f_custom_values, marker='o', linestyle='-', color='b',
             label=f'Custom Function (param = {optimal_param:.4f})')
    plt.plot(a_values, f_exp_values, marker='x', linestyle='--', color='g',
             label=f'Classic Exponential (λ = {lam:.4f})')
    plt.axhline(y=target_f200, color='r', linestyle='--',
                label=f'Target f(200) = {target_f200}')
    plt.title('Comparison of Modified Custom Function and Classic Exponential Function')
    plt.xlabel('Epoch (a)')
    plt.ylabel('f(a)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
