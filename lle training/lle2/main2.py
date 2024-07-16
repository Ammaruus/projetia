
from lle import LLE
import matplotlib

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
display(env)