import lle
from lle import LLE, World
from lle import Action, WorldState
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
#display_world(empty)

# borads with walls
board_with_walls = get_empty_board(6, 5)
board_with_walls[0][0] = "S0"
board_with_walls[4][5] = "X"
board_with_walls[1] = ["@", "@", "@", "@", "@", "."]
board_with_walls[3] = [".", "V", "V", "V", "V", "V"]
save_board(board_with_walls, "walls")
with_walls = load_world("walls")
#display_world(with_walls)

# with gems
board_with_gems = get_empty_board(3, 3)
board_with_gems[0][0] = "S0"
board_with_gems[2][2] = "X"
board_with_gems[2][0] = "G"
board_with_gems[0][2] = "G"
save_board(board_with_gems, "gems")
with_gems = load_world("gems")
#display_world(with_gems)

# with multiple agents
two_agents = get_empty_board(5, 5)
two_agents[0][0] = "S0"
two_agents[4][0] = "S1"
two_agents[4][4] = "X"
two_agents[0][4] = "X"
save_board(two_agents, "two_agents")
world_two_agents = load_world("two_agents")
#display_world(world_two_agents)

# lasers
tatl = get_empty_board(5, 5)
tatl[0][0] = "S0"
tatl[4][0] = "S1"
tatl[4][4] = "X"
tatl[0][4] = "X"
tatl[0][1] = "L0S"
tatl[4][3] = "L1N"
save_board(tatl, "tatl")
world_tatl = load_world("tatl")
#display_world(world_tatl)

# linear
save_board([["S0", "G", "G", "X"]], "linear")
linear = load_world("linear")
#display_world(linear)

# linear with 2 agents
save_board([["S0", "G", ".", "X"], ["S1", ".", "G", "X"]], "linear2")
linear2 = load_world("linear2")
#display_world(linear2)

# complex example
board =[
    ["S0", "@", ".", "G", ".", ".", ".", "G", ".", "."],
    ["S1", "@", ".", ".", ".", ".", ".", ".", ".", "."],
    [".", "@", ".", ".", ".", "G", ".", ".", ".", "."],
    [".", "@", ".", ".", "L0W", "@", "@", "@", "@", "."],
    [".", "@", ".", ".", ".", ".", ".", ".", "V", "."],
    [".", "@", "@", "@", "@", "L1S", "@", ".", "V", "."],
    [".", ".", ".", ".", ".", ".", "@", ".", "V", "."],
    [".", ".", ".", "G", ".", ".", "@", ".", "V", "."],
    [".", ".", ".", "G", ".", ".", "@", ".", "V", "X"],
    ["G", "L0N", ".", ".", ".", ".", ".", ".", "V", "X"]
]
save_board(board, "board")
world = load_world("board")
#display_world(world)


# Playing with agents
#linear.reset()
#display_world(linear)
#events = linear.step([Action.EAST])
#display_world(linear)
#print("Events:", events)

#events = linear.step([Action.EAST])
#display_world(linear)
#print("Events:", events)
#print("Available actions for agent 0:", linear.available_actions()[0])

# getting information about the environment
world.reset()
display_world(world)
print("Current state:", world.get_state())
print("Number of gems:", world.n_gems)
print("Number of collected gems:", world.gems_collected)
print("Position of the walls:", world.wall_pos)
print("Position of the voids:", world.void_pos)
print("Position of the exits:", world.exit_pos)
print("Position of the lasers:", world.lasers)
print("Position of the agents:", world.agents_positions)


