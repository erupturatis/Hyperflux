import numpy as np
import matplotlib.pyplot as plt

from src.schedulers import expected_pruning_decrease_at_epoch, TrajectoryCalculator


def compute_pruning_trajectory(epochs_target, start_decrease, end_decrease, aggressivity_transition, late_aggressivity):
    """
    Computes the pruning factor f for each epoch.

    Parameters:
        epochs_target (int): Total number of epochs.
        start_decrease (float): Initial pruning decrease factor.
        end_decrease (float): Final pruning decrease factor.
        aggressivity_transition (float): Controls the steepness of the sigmoid transition.
        late_aggressivity (float): Epoch at which the pruning aggressiveness starts to increase.

    Returns:
        list of float: Pruning factors for each epoch.
    """
    pruning_factors = []
    cumulative_log = np.log(100)  # Starting with 100%

    for epoch in range(1, epochs_target + 1):
        decrease = expected_pruning_decrease_at_epoch(
            epoch, start_decrease, end_decrease, aggressivity_transition, late_aggressivity
        )
        cumulative_log += np.log(decrease)
        expected_pruning_at_epoch = np.exp(cumulative_log)
        print("epoch:", epoch, "expected_pruning_at_epoch:", expected_pruning_at_epoch)
        pruning_factors.append(expected_pruning_at_epoch)

    return pruning_factors

def plot_pruning_trajectory(pruning_factors, epochs_target, pruning_target):
    """
    Plots the pruning trajectory over epochs.

    Parameters:
        pruning_factors (list of float): Pruning factors for each epoch.
        epochs_target (int): Total number of epochs.
        pruning_target (float): The desired pruning factor at the final epoch.
    """
    epochs = np.arange(1, epochs_target + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, pruning_factors, label='Pruning Factor f(epoch)', color='blue')
    plt.axhline(y=pruning_target, color='red', linestyle='--', label=f'Pruning Target ({pruning_target})')
    plt.title('Pruning Trajectory Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Pruning Factor f(epoch)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define pruning parameters
    pruning_target = 0.5           # Desired pruning factor at the final epoch
    epochs_target = 400            # Total number of epochs

    # Initialize TrajectoryCalculator to find the optimal start_pruning_decrease
    late_aggressivity = epochs_target / 3
    aggressivity_transition = 0.05

    calculator = TrajectoryCalculator(
        pruning_target=pruning_target,
        epochs_target=epochs_target,
        late_aggresivity=late_aggressivity,
        aggresivity_transition=aggressivity_transition,
    )

    start_decrease = calculator.get_start_pruning_decrease()
    end_decrease = calculator.END_PRUNING_DECREASE

    # Compute pruning trajectory
    pruning_factors = compute_pruning_trajectory(
        epochs_target=epochs_target,
        start_decrease=start_decrease,
        end_decrease=end_decrease,
        late_aggressivity=late_aggressivity,
        aggressivity_transition=aggressivity_transition,
    )

    # Plot the pruning trajectory
    plot_pruning_trajectory(pruning_factors, epochs_target, pruning_target)
