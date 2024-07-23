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
actions = [1 , 0, 3, 2]  # Exemple d'actions
game = LabyrinthGame()
game.play(actions)
