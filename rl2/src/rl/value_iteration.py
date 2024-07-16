from .env import Labyrinth


class ValueIteration:
    def __init__(self, gamma: float):
        self.values = ...

    def select_action(self, state: tuple[int, int]) -> int:
        """Renvoie l'action qui maximise la valeur de l'état donné."""

    def train(self, env: Labyrinth, n: int):
        """Entraîne l'agent sur pendant pour n itérations."""
