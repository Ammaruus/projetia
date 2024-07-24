import cv2
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from test import Labyrinth


class LabyrinthGame:
    def __init__(self):
        self.env = Labyrinth()
        self.visited_positions = []

    def play(self, actions):
        self.env.reset()
        self.env.render()
        self.visited_positions.append(self.env.get_observation())
        time.sleep(2)  # Pause initiale pour le premier rendu
        for action in actions:
            if self.env.is_done():
                print("Le jeu est terminé.")
                break
            print("actions possible de l'agent :",self.env._world.available_actions())
            self.env.step(action)
            self.env.render()
            self.env.render()
            self.visited_positions.append(self.env.get_observation())
            time.sleep(1)  # Pause de 2 secondes après chaque mouvement
        cv2.destroyAllWindows()


# Actions: 0 - Nord, 1 - Est, 2 - Sud, 3 - Ouest
actions = [1 , 0, 3, 2, 0, 1, 2, 3, 0, 1, 2, 3]  # Exemple d'actions
game = LabyrinthGame()
game.play(actions)