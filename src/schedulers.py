
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
        if current_decrease > expected_decrease:
            print("Baseline increased !!")
            # expected deviation
            self.baseline += 1 + self.streak
            self.streak += 0.5
        else:
            self.streak = 0
            self.streak -= 0.1

    def get_multiplier(self) -> int:
        return self.baseline ** self.exponent_constant
