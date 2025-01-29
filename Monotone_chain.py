# This is the final script for part b, specifically the Monotone chain algorithm

import numpy as np
import matplotlib.pyplot as plt

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def read_points_from_file(filename):
    # Once again, since pulling data from mesh.dat, skipping first row of input file
    data = np.loadtxt(filename, skiprows=1)  
    return [Point(x, y) for x, y in data]

def orientation(p, q, r):
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if val == 0:
        return 0 
    elif val > 0:
        return 1  
    else:
        return 2 

def monotone(points):
    # Locating laterally extrema
    points = sorted(points, key=lambda p: (p.x, p.y))
    if len(points) < 3:
        return points

    lower = []
    for p in points:
        while len(lower) >= 2 and orientation(lower[-2], lower[-1], p) != 2:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and orientation(upper[-2], upper[-1], p) != 2:
            upper.pop()
        upper.append(p)

    # Since the hull construction separates the upper and lower halves, 
    # there is point redundancy that must be accounted for.
    del lower[-1]
    del upper[-1]

    # Combination of upper and lower hulls.
    hull = lower + upper

def printHull(hull):
    print("The points in Convex Hull are:")
    for p in hull:
        print(f"({p.x}, {p.y}) ", end="")

if __name__ == "__main__":
    filename = 'mesh.dat'
    points = read_points_from_file(filename)
    hull = monotone(points)
    printHull(hull)

    # For visualization
    all_points = np.array([[p.x, p.y] for p in points])
    hull_points = np.array([[p.x, p.y] for p in hull] + [[hull[0].x, hull[0].y]])

    plt.scatter(all_points[:, 0], all_points[:, 1], label='Points')
    plt.plot(hull_points[:, 0], hull_points[:, 1], 'r-', label='Convex Hull')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Convex Hull using Monotone Chain")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('Monotone_chain_mesh.png')

# Part c of the first task is to operate the convex hull functions on the mesh.dat
# file, though in order to check the efficacy of the program I used the mesh.dat
# as I constructed each one of the scripts. In other words, part c of task 1 is embedded
# in the programs written for each of the convex hulls. This allows me to move on to task 2.