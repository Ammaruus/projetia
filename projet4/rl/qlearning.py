from .env import Labyrinth


class QLearning:
    def __init__(self, gamma: float, alpha: float, epsilon: float, n_actions: int):
        self.q_table = ...

    def select_action(self, state: tuple[int, int]) -> int:
        """Choisit une action à partir de l'état donné"""

    def train(self, env: Labyrinth, n_steps: int):
        """Entraîne l'agent pendant n_steps étapes, et renvoie la liste des scores obtenus par l'agent au cours de l'entraînement."""
