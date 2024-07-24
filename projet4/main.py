from rl.env import Labyrinth
from rl.qlearning import QLearning
from itertools import product
from rl.value_iteration import ValueIteration
import lle

if __name__ == "_main":
    env = Labyrinth()
    vi = ValueIteration(env, gamma=0.9)  # Instantiate ValueIteration with a discount factor gamma
    while True:
        env.reset()
        env.render()
        print(env._world.available_actions())
        print(env.get_observation())
        #print(env.step(4))
        print(vi.get_next_state(env.get_observation(), lle.Action(4)))
        #print(vi.select_action(env.get_observation()))

        input()
    #algo = QLearning()
    #algo.train(env, 20_000)
