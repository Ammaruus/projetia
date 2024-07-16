
from lle import LLE, Action
import matplotlib
import rlenv
import numpy as np
import random

# Changer le backend de Matplotlib
matplotlib.use('TkAgg')  # Ou 'Qt5Agg' ou 'GTK3Agg'
import matplotlib.pyplot as plt

DISPLAY = True

def display(env: LLE):
    plt.imshow(env.render("rgb_array"))
    plt.axis('off')
    plt.show()

# for lvl in [4, 5, 6]:
#     env = LLE.level(lvl)
#     display(env)

env = LLE.from_str(
    """
S0 . . G
 @ . . L0W
 . G . S1
 X . . X
"""
)
#display(env)

# observations
# initial_observation = env.reset()
# print("Observation shape", initial_observation.data.shape)
# print("Extras shape", initial_observation.extras_shape)
# print("State shape", initial_observation.state.shape)
# print("Available actions:\n", initial_observation.available_actions)

env = (rlenv.Builder(LLE.from_str(
    """
S0 . . G
 @ . . L0W
 . G . S1
 X . . X
"""
    ))
    .agent_id()
    .time_limit(5)
    .build())
#display(env)
# obs = env.reset()
# # print("Observation shape", obs.data.shape)
# # print("Extras with agent ID", obs.extras)
# # print("State shape", obs.state.shape)
# # print("Available actions:\n", obs.available_actions)

# step_data = env.step([Action.EAST.value, Action.WEST.value])
# # unpack the tuple
# obs, reward, done, truncated, info = step_data

# display(env)


# Action loop
terminated = False
obs = env.reset()
while not terminated:
    available_actions = [np.nonzero(available)[0] for available in obs.available_actions]
    actions = [random.choice(agent_actions) for agent_actions in available_actions]
    obs, r, done, truncated, info = env.step(actions)
    terminated = done
    display(env)

