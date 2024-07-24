from env import Labyrinth
from mdp import MDP
from env_mdp import ENVMDP

from typing import TypeVar
import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

A = TypeVar("A")
S = TypeVar("S")

class ValueIteration:
    def __init__(self, mdp: MDP, gamma: float):
        self.gamma = gamma
        self.mdp = mdp
        # Initialize all state values
        self.values = {state: 0.0 for state in mdp.states()}
        # Initialize visited positions tracker
        self.visited_positions = []

    def value(self, state: S) -> float:
        """Returns the value of the given state."""
        if state not in self.values:
            return 0.0
        return self.values.get(state)

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

    def train(self, env: Labyrinth, n: int, iterations: int):
        """Entraîne l'agent pour n itérations multiples fois."""
        for _ in range(iterations):
            env.reset()
            self.value_iteration(n)
            for state in self.mdp.states():
                self.visited_positions.append(state.agents_positions[0])
        self.plot_heatmap(env)

    def plot_heatmap(self, env: Labyrinth):
        # Create a matrix for visited positions
        height, width = env._world.height, env._world.width
        heatmap = np.zeros((height, width))

        for pos in self.visited_positions:
            heatmap[pos[0], pos[1]] += 1

        # Use seaborn to create the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap, annot=True, cmap="YlGnBu", cbar=True)
        plt.title("Heatmap des positions visitées dans le labyrinthe")
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
    #print(f"Initial value for the initial state: {vi.value(initial_state)}")
    
    # Perform value iteration for a number of iterations
    n_iterations = 10
    
    # Train the agent 10 times
    vi.train(env, n_iterations, 10)
    
    # Test the updated value method
    updated_value = vi.value(initial_state)
    #print(f"Updated value for the initial state after {n_iterations * training_iterations} iterations: {updated_value}")
    
    # Test the select_action method
    best_action = vi.select_action(initial_state)
    print(f"Best action for the initial state: {best_action}")
    

if __name__ == "__main__":
    test_value_iteration()
