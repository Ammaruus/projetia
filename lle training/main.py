import lle
from lle import LLE, World
import matplotlib

# Changer le backend de Matplotlib
matplotlib.use('TkAgg')  # Ou 'Qt5Agg' ou 'GTK3Agg'
import matplotlib.pyplot as plt

DISPLAY = True

# helper functions

def get_empty_board(x, y):
    return [["." for _ in range(x)] for _ in range(y)]

def get_board_as_str(board):
    return "\n".join([" ".join(i) for i in board])

def save_board(board, filename):
    """Saves a board to a file in order to reload it later"""
    f = open("./boards/"+filename, "w")
    f.write(get_board_as_str(board))
    f.close()

def load_world(filename):
    return lle.World.from_file("./boards/"+filename)

def display_world(w):
    if DISPLAY:
        plt.imshow(w.get_image())
        plt.axis('off')
        plt.show()

# code 
empty_board = get_empty_board(6, 5)
empty_board[0][0] = "S0"
empty_board[4][5] = "X"
save_board(empty_board, "empty")

# load the world
empty = load_world("empty")
display_world(empty)
