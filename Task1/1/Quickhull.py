# This was the convex hull method that I could not get working the first time, though now I have gotten
# it to work finally. See the image in the directory for final result.

import numpy as np
import matplotlib.pyplot as plt

def quickhull(points):
    def find_side(p1, p2, p):

        return (p[0] - p1[0]) * (p2[1] - p1[1]) - (p[1] - p1[1]) * (p2[0] - p1[0])

    def distance(p1, p2, p):
        
        return np.abs(find_side(p1, p2, p)) / np.linalg.norm(p2-p1)

    def farthest_point(p1, p2, points):
        
        max_distance = 0
        farthest = None
        for p in points:
            dist = distance(p1, p2, p)
            if dist > max_distance:
                max_distance = dist
                farthest = p
        return farthest

    def hull_recursion(p1, p2, points):
    
        if not len(points):
            return []
        
        fp = farthest_point(p1, p2, points)
        if fp is None:
            return []

        points_left_to_p1_fp = [p for p in points if find_side(p1, fp, p) > 0]
        points_left_to_fp_p2 = [p for p in points if find_side(fp, p2, p) > 0]

        return hull_recursion(p1, fp, points_left_to_p1_fp) + [fp] + hull_recursion(fp, p2, points_left_to_fp_p2)

    min_point = min(points, key=lambda p: p[0])
    max_point = max(points, key=lambda p: p[0])
    points_above = [p for p in points if find_side(min_point, max_point, p) > 0]
    points_below = [p for p in points if find_side(min_point, max_point, p) < 0]
    upper_hull = hull_recursion(min_point, max_point, points_above)
    lower_hull = hull_recursion(max_point, min_point, points_below)
    full_hull = [min_point] + upper_hull + [max_point] + lower_hull + [min_point] 
    hull_points = np.array(full_hull)
    plt.figure(figsize=(10, 6))
    plt.scatter(points[:, 0], points[:, 1], color='blue')
    plt.plot(hull_points[:, 0], hull_points[:, 1], 'r-o', linewidth=2)
    plt.title("Convex Hull via Quickhull")
    plt.show()
    plt.savefig('Quickhull_mesh.png')

    return hull_points

points = np.loadtxt('mesh.dat', skiprows=1)
quickhull(points)


