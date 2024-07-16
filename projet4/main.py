from rl.env import Labyrinth
from rl.qlearning import QLearning

if __name__ == "__main__":
    env = Labyrinth()
    while True:
        env.reset()
        env.render()
        input()

    algo = QLearning()
    algo.train(env, 20_000)
