from test import Labyrinth
from mdp import MDP
from env_mdp import ENVMDP

from typing import Tuple
from typing import TypeVar, Generic
from abc import abstractmethod, ABC
import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lle

A = TypeVar("A")
S = TypeVar("S")

class ValueIteration:
    def __init__(self, mdp: MDP, gamma: float):
        self.gamma = gamma
        self.mdp = mdp
        # Initialize all state values
        self.values = {state: 0.0 for state in mdp.states()}

    def value(self, state: S) -> float:
        """Returns the value of the given state."""
        if state not in self.values:
            return 0.0
        return self.values.get(state)   
    
    def availabale_action(state: Tuple[int,int]) -> list[int]:
        return Labyrinth._world.available_actions()

    def select_action(self, state: tuple[int, int]) -> int:
        """Renvoie l'action qui maximise la valeur de l'état donné."""
        available_actions = self.mdp.available_actions(state)
        if not available_actions:
            print("No available actions for state", state)
            return None  # Or some default action if appropriate
        return max(available_actions, key=lambda action: self.qvalue(state, action))
    
    def qvalue(self, state: S, action: A) -> float:
        """
        Returns the Q-value of the given state-action pair based on the state values.
        from Bellman equation: Q(s,a) = Sum(P(s,a,s') * (R(s,a,s') + gamma * V(s')))
        """
        qvalue = 0.0
        next_states_and_probs = self.mdp.transitions(state, action)
        for next_state, prob in next_states_and_probs:
            reward = self.mdp.reward(state, action, next_state)
            next_state_value = self.value(next_state)
            qvalue += prob * (reward + self.gamma * next_state_value)
        return qvalue

    def _compute_value_from_qvalues(self, state: S) -> float:
        """
        Returns the value of the given state based on the Q-values.
        from Bellman equation: V(s) = max_a Sum(P(s,a,s') * (R(s,a,s') + gamma * V(s')))
        """
        value = max(
            self.qvalue(state, action) for action in self.mdp.available_actions(state)
        )
        if value is None:
            return 0.0
        return value
    
    def get_values_at_position(self, i: int, j: int) -> list[float]:
        """Returns the values of the states at the given position."""
        states_at_position = [
            state for state in self.mdp.states() if state.agents_positions[0] == (i, j)
        ]
        values_at_position = [self.value(state) for state in states_at_position]
        increasing_values = sorted(values_at_position)

        return increasing_values

    def print_values_table(self, n: int = 0):
        """In a map's representation table, each tile contains the possible values at that position."""
        if not isinstance(self.mdp, ENVMDP):
            return None
        print("Iteration", n, "Values table: ")
        max_len = 0
        for i in range(self.mdp.world.height):
            for j in range(self.mdp.world.width):
                values = self.get_values_at_position(i, j)
                values_str = str(values)
                max_len = max(max_len, len(values_str))

        for i in range(self.mdp.world.height):
            for j in range(self.mdp.world.width):
                values = self.get_values_at_position(i, j)
                print(f"{str(values):<{max_len}}", end=" ")
            print()

    def print_iteration_values(self, iteration: int):
        """Prints the states and their values."""
        for state in self.mdp.states():
            print(state, self.value(state))

    def value_iteration(self, n: int):
        """Performs value iteration for the given number of iterations."""
        for _ in range(n):
            new_values = copy.deepcopy(self.values)
            for state in self.mdp.states():
                if self.mdp.is_final(state):
                    new_values[state] = 0.0
                else:
                    new_values[state] = self._compute_value_from_qvalues(state)
            self.values = new_values
            #self.print_values_table(_)
    
    def get_next_state(self, state: Tuple[int, int], action: int) -> (Tuple[int, int], float):
        """Returns the next state given a state and an action."""
        labyrinth = Labyrinth()
        labyrinth._world.set_state(lle.WorldState([state], []))
        rewards = labyrinth.step(4)
        print("aaa")
        return labyrinth.get_observation()

    def train(self, env: Labyrinth, iterations: list[int]):
        """Entraîne l'agent pour les étapes d'entraînement spécifiées et affiche les heatmaps."""
        for n in iterations:
            self.value_iteration(n)
            self.plot_heatmap(env, n)
    
    def get_values_grid(self) -> np.ndarray:
        """Returns a grid of state values for plotting."""
        height, width = self.mdp.world.height, self.mdp.world.width
        values_grid = np.zeros((height, width))
        for state in self.mdp.states():
            i, j = state.agents_positions[0]
            values_grid[i, j] = self.value(state)
        return values_grid

    # afficher la map de l'entrainement
    def plot_heatmap(self, env: Labyrinth, iteration: int):
        """Plot the heatmap of state values."""
        values_grid = self.get_values_grid()

        plt.figure(figsize=(10, 8))
        sns.heatmap(values_grid, annot=True, cmap="YlGnBu", cbar=True)
        plt.title(f"Heatmap des valeurs des états après {iteration} étapes d'entraînement")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.gca().invert_yaxis()  # Inverser l'axe Y pour correspondre à la représentation graphique
        plt.show()
def test_value_iteration():
    # Initialize the environment
    env = Labyrinth()
    
    # Initialize the MDP
    mdp = ENVMDP(env._world)
    
    # Initialize the Value Iteration algorithm with gamma = 0.9
    gamma = 0.9
    vi = ValueIteration(mdp, gamma)
    
    # Test the value method
    initial_state = env._world.get_state()
    #initial_position = env.get_initial_position()
    #initial_state = lle.WorldState([initial_position], [])
    print(f"Initial state: {initial_state}")
    print(f"Initial value for the initial state: {vi.value(initial_state)}")
    
    # Perform value iteration for specific numbers of iterations
    iteration_steps = [1,2, 3, 4, 5]
    
    
    # Train the agent and plot heatmaps
    vi.train(env, iteration_steps)
    
    # Test the updated value method
    updated_value = vi.value(initial_state)
    print(f"Updated value for the initial state after {iteration_steps[-1]} iterations: {updated_value}")
    
    # Test the select_action method
    best_action = vi.select_action(initial_state)
    print(f"Best action for the initial state: {best_action}")


if __name__ == "__main__":
    test_value_iteration()
