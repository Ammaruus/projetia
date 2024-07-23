from .env import Labyrinth
import numpy as np


class ValueIteration:
    def __init__(self, gamma: float):
        self.values = None
        self.gamma = gamma

    def select_action(self, state: tuple[int, int]) -> int:
        """Renvoie l'action qui maximise la valeur de l'état donné."""
        best_action = None
        best_value = float('-inf')
        for action in range(4):  # Supposons qu'il y a 4 actions possibles
            next_state, reward = env.transition(state, action)
            value = reward + self.gamma * self.values[next_state]
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    def train(self, env: Labyrinth, n: int):
        """Entraîne l'agent sur pendant pour n itérations."""
