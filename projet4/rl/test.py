import cv2
import random
from itertools import product
import lle
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Labyrinth:
    def __init__(self):
        self._world = lle.World("""
.  . . . . . .
S0 . . . . . .
.  . . . . . .
.  . . . . . .
.  . . . . . .
.  . . . . . .
.  . . . . . X""")
        self._done = False
        self._first_render = True
        all_positions = set(product(range(self._world.height), range(self._world.width)))
        self._start_positions = list(all_positions - set(self._world.wall_pos) - set(self._world.exit_pos))

    def reset(self):
        self._done = False
        self._world.reset()
        initial_position = random.choice(self._start_positions)
        self._world.set_state(lle.WorldState([initial_position], []))

    def step(self, action: int) -> float:
        """Effectue une action sur l'environnement et retourne la récompense associée"""
        result = self._world.step([lle.Action(action)])
        
        if isinstance(result, float):
            events = []
            reward = result
        else:
            events, reward = result, -1.0  # Assumant une valeur par défaut si nécessaire

        for event in events:
            if event.event_type == lle.EventType.AGENT_EXIT:
                self._done = True
                return 0.0
        return reward

    def get_observation(self) -> tuple[int, int]:
        return self._world.agents_positions[0]

    def is_done(self) -> bool:
        return self._done

    def render(self):
        img = self._world.get_image()
        if self._first_render:
            # Résout un bug tel que le premier rendu ne s'affiche pas correctement la première fois
            cv2.imshow("Labyrinth", img)
            cv2.waitKey(1)
            self._first_render = False
            import time
            time.sleep(0.2)

        cv2.imshow("Labyrinth", img)
        cv2.waitKey(1)



