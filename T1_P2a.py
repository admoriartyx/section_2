# Need to write program to generate 2D point cloud with n points,
# all coordinates are between 0 and 1.
# Luckily for us, np.random.rand() does exactly this.

import numpy as np
import matplotlib.pyplot as plt

def n_point_cloud(n):
    
    points = np.random.rand(n, 2)
    x, y = points[:, 0], points[:, 1]
    plt.scatter(x, y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"{n}-Point Cloud Generation")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()
    plt.savefig(f'{n}_point_cloud.png')

# I will start off with a 10 point test.

n_point_cloud(10)

# The test was successful. Now moving onto part b in another .py source.