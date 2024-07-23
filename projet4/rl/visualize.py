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
        self.plot_heatmap()

    def plot_heatmap(self):
        # Créer une matrice pour les positions visitées
        height, width = self.env._world.height, self.env._world.width
        heatmap = np.zeros((height, width))

        for pos in self.visited_positions:
            heatmap[pos[0], pos[1]] += 1

        # Utiliser seaborn pour créer la heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap, annot=True, cmap="YlGnBu", cbar=True)
        plt.title("Heatmap des positions visitées dans le labyrinthe")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.gca().invert_yaxis()  # Inverser l'axe Y pour correspondre à la représentation graphique
        plt.show()


# Actions: 0 - Nord, 1 - Est, 2 - Sud, 3 - Ouest
actions = [1 , 0, 3, 2, 0, 1, 2, 3, 0, 1, 2, 3]  # Exemple d'actions
game = LabyrinthGame()
game.play(actions)