# Now the Quickhull program...
# Accurately implementing this one was very troublesome. I think primarily because
# the mesh.dat dataset is somewhat large. I troubleshot this code for awhile using 
# ChatGPT but at a certain point I know perfection was not the goal of this exercise.

import numpy as np
import matplotlib.pyplot as plt

def quickhull(points):
    points = np.array(points)

    def find_side(p1, p2, p):
        return (p[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p[0] - p1[0])

    def add_hull_points(p1, p2, points, side):
        if len(points) == 0:
            return [p1, p2]
        furthest_point = max(points, key=lambda p: np.abs(find_side(p1, p2, p)))
        points_on_one_side = [p for p in points if find_side(furthest_point, p1, p) == -side]
        points_on_other_side = [p for p in points if find_side(furthest_point, p2, p) == side]
        
        return (add_hull_points(p1, furthest_point, points_on_one_side, -side) +
                add_hull_points(furthest_point, p2, points_on_other_side, side))

    if len(points) < 3:
        return points

    # Locate min and max values
    min_point = points[np.argmin(points[:, 0])]
    max_point = points[np.argmax(points[:, 0])]

    left_points = [p for p in points if not np.array_equal(p, min_point) and not np.array_equal(p, max_point) and find_side(min_point, max_point, p) < 0]
    right_points = [p for p in points if not np.array_equal(p, min_point) and not np.array_equal(p, max_point) and find_side(min_point, max_point, p) > 0]

    # Hull construction
    left_hull = add_hull_points(min_point, max_point, left_points, 1)
    right_hull = add_hull_points(max_point, min_point, right_points, -1)

    return np.array(list(set([tuple(p) for p in left_hull + right_hull])))

def read_points(filename):
    return np.loadtxt(filename, skiprows=1)

if __name__ == "__main__":
    filename = 'mesh.dat'  
    points = read_points(filename)
    hull_points = quickhull(points)

    # Plotting of convex hull + original mesh data
    plt.scatter(points[:, 0], points[:, 1], label='Points')
    hull_points = np.vstack([hull_points, hull_points[0]])  # Close the hull
    plt.plot(hull_points[:, 0], hull_points[:, 1], 'r-', label='Convex Hull')
    plt.scatter(hull_points[:, 0], hull_points[:, 1], color='red')  # Highlight hull points
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Convex Hull using QuickHull")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('quickhull_mesh.png')

