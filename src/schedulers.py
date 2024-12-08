from typing import Dict
import numpy as np


class PruningScheduler:
    def __init__(self, exponent_constant: float, pruning_target: float, epochs_target: int, total_parameters: int):
        self.baseline = 0
        self.exponent_constant = exponent_constant
        self.pruning_target = pruning_target
        self.epochs_target = epochs_target
        self.total_parameters = total_parameters
        self.streak = 0

        self.recorded_states = []

    def record_state(self, remaining_weights: float):
        """
        Appends current number of remaining weights to the list of recorded states
        """
        self.recorded_states.append(remaining_weights)


    def get_previous_decrease(self) -> float:
        """
        Get the percentage decrease from the previous state to the current state
        """
        if len(self.recorded_states) < 2:
            return -1

        if len(self.recorded_states) == 2:
            return self.recorded_states[-1] / self.recorded_states[-2]

        return ((self.recorded_states[-1] / self.recorded_states[-2]) + (self.recorded_states[-2] / self.recorded_states[-3]))/ 2

    def get_expected_percentage_decrease(self) -> float:
        """
        current * (percentage ** remaining_epochs) = desired
        percentage = (desired / current) ** (1 / remaining_epochs)
        """
        desired_remaining_parameters = self.total_parameters * self.pruning_target
        remaining_epochs = self.epochs_target - len(self.recorded_states)
        current_parameters = self.recorded_states[-1]
        return (desired_remaining_parameters / current_parameters) ** (1 / remaining_epochs)


    def get_remaining_epochs(self) -> int:
        return self.epochs_target - len(self.recorded_states)

    def step(self) -> None:
        """
        Attempts to predict the final number of weights that will remain after the pruning process, given current pace

        Formula for decreases
        params * decrease = pruned params
        So 0.8 is more aggressive than 0.9
        """
        if self.get_remaining_epochs() <= 0:
            self.baseline = 0
            return None

        current_decrease = self.get_previous_decrease()
        if current_decrease == -1:
            print("Not enough data to predict")
            return None

        expected_decrease = self.get_expected_percentage_decrease()

        desired_remaining_parameters = self.total_parameters * self.pruning_target
        remaining_epochs = self.epochs_target - len(self.recorded_states)

        print(f"Current decrease: {current_decrease * 100:.2f}%, Expected decrease: {expected_decrease * 100:.2f}%")
        if current_decrease > expected_decrease and expected_decrease < 1:
            print("Baseline increased !!")
            # expected deviation
            self.baseline += 0.3 + self.streak
            self.streak += 0.1
        else:
            self.streak = 0
            self.baseline -= 0.15
            if self.baseline < 0:
                self.baseline = 0

    def get_multiplier(self) -> int:
        return self.baseline ** self.exponent_constant

class PruningSchedulerSane:
    def __init__(self, exponent_constant: float, pruning_target: float, epochs_target: int, total_parameters: int):
        self.baseline = 0
        self.exponent_constant = exponent_constant
        self.pruning_target = pruning_target * 100 # 100 for the sake of compatibility with the other scheduler
        self.epochs_target = epochs_target
        self.total_parameters = total_parameters
        self.streak = 0
        self.recorded_states = []
        self.epoch = 0

        late_aggressivity = epochs_target / 3
        aggressivity_transition = 0.05
        # calculate target trajectory
        self.trajectory_calculator = TrajectoryCalculator(
            pruning_target=self.pruning_target,
            epochs_target=self.epochs_target,
            late_aggresivity=late_aggressivity,
            aggresivity_transition=aggressivity_transition
        )

    def record_state(self, remaining_weights: int):
        """
        Recorded states represent the number of params at the end of epoch self.epoch
        """
        print(f"Remaining weights: {remaining_weights}")
        self.recorded_states.append(remaining_weights)
        self.epoch += 1

    def _get_expected_parameters_percentage(self) -> float:
        return self.trajectory_calculator.get_expected_pruning_at_epoch(self.epoch)

    def _get_current_parameters_percentage(self) -> float:
        return self.recorded_states[-1] / self.total_parameters * 100

    def step(self) -> None:
        """
        Adjusts basline such that network params match expected params
        """

        # we are at the end of this epoch
        if self.epoch > self.epochs_target:
            self.baseline = 0
            return None

        expected_params = self._get_expected_parameters_percentage()
        current_params = self._get_current_parameters_percentage()

        print("Current status normalized: Current params", current_params , "Expected params", expected_params )

        if current_params > expected_params:
            # network has too many params, prune more aggresive
            print("Baseline increased !!")
            # expected deviation
            self.baseline += 0.3 + self.streak
            self.streak += 0.1
        else:
            # Ease up presssure
            self.streak = 0
            self.baseline -= 0.15
            if self.baseline < 0:
                self.baseline = 0

    def get_multiplier(self) -> int:
        return self.baseline ** self.exponent_constant


def expected_pruning_decrease_at_epoch(epoch, start_decrease, end_decrease, aggressivity_transition, late_aggressivity):
    """
    Calculates the expected pruning decrease at a given epoch using a sigmoid function.

    Parameters:
        epoch (int): Current epoch.
        start_decrease (float): Initial pruning decrease factor.
        end_decrease (float): Final pruning decrease factor.
        aggressivity_transition (float): Controls the steepness of the sigmoid transition.
        late_aggressivity (float): Epoch at which the pruning aggressiveness starts to increase.

    Returns:
        float: Pruning decrease factor at the given epoch.
    """
    sigmoid = 1 / (1 + np.exp(-aggressivity_transition * (epoch - late_aggressivity)))
    return start_decrease + (end_decrease - start_decrease) * sigmoid

def expected_pruning(epochs_target, start_decrease, end_decrease, aggressivity_transition, late_aggressivity):
    """
    Calculates the cumulative pruning factor up to the target epoch.

    Parameters:
        epochs_target (int): Total number of epochs.
        start_decrease (float): Initial pruning decrease factor.
        end_decrease (float): Final pruning decrease factor.
        aggressivity_transition (float): Controls the steepness of the sigmoid transition.
        late_aggressivity (float): Epoch at which the pruning aggressiveness starts to increase.

    Returns:
        float: Cumulative pruning factor at the target epoch.
    """
    log_e_values = [
        np.log(expected_pruning_decrease_at_epoch(
            epoch, start_decrease, end_decrease, aggressivity_transition, late_aggressivity
        ))
        for epoch in range(1, epochs_target + 1)
    ]
    log_product = np.sum(log_e_values)
    # 100 is the initial parameter count (or 100%)
    log_fa = np.log(100) + log_product
    return np.exp(log_fa)

class TrajectoryCalculator:
    def __init__(self, pruning_target, epochs_target, late_aggresivity, aggresivity_transition):
        """
        Initializes the TrajectoryCalculator with the desired pruning parameters.
        Parameters:
            pruning_target (float): The desired pruning factor at the final epoch.
            epochs_target (int): Total number of epochs.
            late_aggressivity (float): Epoch at which the pruning aggressiveness starts to increase.
            aggressivity_transition (float): Controls the steepness of the sigmoid transition.
        """
        self.pruning_target = pruning_target
        self.epochs_target = epochs_target

        self.late_aggressivity = late_aggresivity
        self.aggressivity_transition = aggresivity_transition

        self.END_PRUNING_DECREASE = 0.999
        self.start_pruning_decrease = 0.0

        self.find_start_pruning_decrease()

    def find_start_pruning_decrease(self):
        """
        Finds the optimal starting pruning decrease factor using binary search to meet the pruning target.
        """
        lower_start = 0.0
        upper_start = 0.999

        iteration = 0
        max_iterations = 100

        best_start_decrease = None
        best_end_pruning = None
        MARGIN_ERROR = 1e-6

        while iteration < max_iterations:
            mid_start = (lower_start + upper_start) / 2
            current_end_pruning = expected_pruning(
                self.epochs_target,
                mid_start,
                self.END_PRUNING_DECREASE,
                self.aggressivity_transition,
                self.late_aggressivity
            )
            # print(f"Iteration {iteration+1}: start_decrease={mid_start:.6f}, f({self.epochs_target})={current_end_pruning:.6f}")

            if abs(current_end_pruning - self.pruning_target) < MARGIN_ERROR:
                best_start_decrease = mid_start
                best_end_pruning = current_end_pruning
                break

            if current_end_pruning < self.pruning_target:
                lower_start = mid_start
            else:
                upper_start = mid_start

            if best_end_pruning is None or abs(current_end_pruning - self.pruning_target) < abs(best_end_pruning - self.pruning_target):
                best_start_decrease = mid_start
                best_end_pruning = current_end_pruning

            iteration += 1

        self.start_pruning_decrease = best_start_decrease
        print(f"\nOptimal start_pruning_decrease: {self.start_pruning_decrease:.6f}")
        print(f"Final pruning at epoch {self.epochs_target}: {best_end_pruning:.6f}")

    def get_start_pruning_decrease(self):
        """
        Returns the optimal starting pruning decrease factor.

        Returns:
            float: Optimal start pruning decrease.
        """
        return self.start_pruning_decrease

    def get_expected_pruning_at_epoch(self, epoch:int):
        return expected_pruning(
            epochs_target=epoch,
            start_decrease=self.start_pruning_decrease,
            end_decrease=self.END_PRUNING_DECREASE,
            aggressivity_transition=self.aggressivity_transition,
            late_aggressivity=self.late_aggressivity
        )

