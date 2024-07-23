import numpy as np
from .env import Labyrinth

class ValueIteration:
    def __init__(self, gamma: float):
        self.values = {}  # Dictionary to store state values
        self.gamma = gamma

    def select_action(self, state: tuple[int, int]) -> int:
        """Returns the action that maximizes the value of the given state."""
        available_actions = self.get_available_actions(state)
        if not available_actions:
            print("No available actions for state", state)
            return None  # Or some default action if appropriate
        return max(available_actions, key=lambda action: self.qvalue(state, action))

    def qvalue(self, state: tuple[int, int], action: int) -> float:
        """
        Returns the Q-value of the given state-action pair
        based on the state values.
        """
        qvalue = 0.0
        next_states_and_probs = self.get_transitions(state, action)
        for next_state, prob in next_states_and_probs:
            reward = self.get_reward(state, action, next_state)
            next_state_value = self.values.get(next_state, 0.0)
            qvalue += prob * (reward + self.gamma * next_state_value)
        return qvalue

    def _compute_value_from_qvalues(self, state: tuple[int, int]) -> float:
        """
        Returns the value of the given state based on the Q-values.
        """
        available_actions = self.get_available_actions(state)
        if not available_actions:
            return 0.0
        value = max(self.qvalue(state, action) for action in available_actions)
        return value

    def get_values_at_position(self, i: int, j: int) -> list[float]:
        """Returns the values of the states at the given position."""
        states_at_position = [(x, y) for (x, y) in self.values.keys() if (x, y) == (i, j)]
        values_at_position = [self.values.get(state, 0.0) for state in states_at_position]
        increasing_values = sorted(values_at_position)
        return increasing_values

    def print_values_table(self, n: int = 0):
        """Prints a table of values at each position in the Labyrinth."""
        print("Iteration", n, "Values table: ")
        max_len = 0
        for i in range(self.env.height):
            for j in range(self.env.width):
                values = self.get_values_at_position(i, j)
                values_str = str(values)
                max_len = max(max_len, len(values_str))

        for i in range(self.env.height):
            for j in range(self.env.width):
                values = self.get_values_at_position(i, j)
                print(f"{str(values):<{max_len}}", end=" ")
            print()

    def print_iteration_values(self, iteration: int):
        """Prints the states and their values."""
        for state in self.values:
            print(state, self.values[state])

    def train(self, env: Labyrinth, n: int):
        """Trains the agent for n iterations."""
        self.env = env
        for _ in range(n):
            new_values = self.values.copy()
            for state in self.env.get_all_states():
                if self.env.is_final(state):
                    new_values[state] = 0.0
                else:
                    new_values[state] = self._compute_value_from_qvalues(state)
            self.values = new_values
            self.print_values_table(_)

    def get_available_actions(self, state: tuple[int, int]) -> list[int]:
        """Returns a list of available actions for the given state."""
        return self.env.get_actions(state)

    def get_transitions(self, state: tuple[int, int], action: int) -> list[tuple[tuple[int, int], float]]:
        """Returns a list of (next_state, probability) pairs for a given state-action pair."""
        return self.env.get_transitions(state, action)

    def get_reward(self, state: tuple[int, int], action: int, next_state: tuple[int, int]) -> float:
        """Returns the reward for transitioning from state to next_state with action."""
        return self.env.get_reward(state, action, next_state)
