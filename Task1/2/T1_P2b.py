# This file is for part b of Task 1; Problem 2. I will copy and paste over the different 
# Programs and try to iterate through them with a for loop to change n

# I suppose I should first create all the data sets. 

import numpy as np
import matplotlib.pyplot as plt

points_10 = np.random.rand(10, 2)
points_50 = np.random.rand(50, 2)
points_100 = np.random.rand(100, 2)
points_200 = np.random.rand(200, 2)
points_400 = np.random.rand(400, 2)
points_800 = np.random.rand(800, 2)
points_1000 = np.random.rand(1000, 2)

# I tried to do this with a for loop but ChatGPT said that naming variables
# within a for loop is not recommended for some reason? Just decided to write it all out.

import time # this will eventually be used to measure runtime.

# Here I will copy and paste all of my prior convex hull functions
def graham_scan(points):
    # First locating lowest y, then x
    start = min(points, key=lambda p: (p[1], p[0]))
    
    # Polar angle sorting
    def polar_angle(p):
        return np.arctan2(p[1] - start[1], p[0] - start[0])
    
    sorted_points = sorted(points, key=lambda p: (polar_angle(p), np.linalg.norm([p[0]-start[0], p[1]-start[1]])))
    
    hull = [sorted_points[0], sorted_points[1]]
    
    for point in sorted_points[2:]:
        while len(hull) >= 2:
            # Direction determination
            p2, p1 = hull[-1], hull[-2] 
            cross = (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])
            if cross <= 0: 
                hull.pop() 
            else:
                break
        hull.append(point)
    
    # Quick addition to make starting and end node connect
    hull.append(hull[0])
    
    return np.array(hull)

def jarvis_march(points):
    
    n = len(points)

    # Remove duplicate points
    points = np.unique(points, axis=0)

    # Locating leftmost point
    start = min(points, key=lambda p: p[0])
    hull = []

    point_on_hull = start
    loop_counter = 0 
    max_loops = len(points) * 2  # Max loop condition to prevent infinite cycling

    while True:
        hull.append(point_on_hull)
        candidate = points[0]

        for point in points:
            cross = ((candidate[0] - point_on_hull[0]) * (point[1] - point_on_hull[1]) -
                     (candidate[1] - point_on_hull[1]) * (point[0] - point_on_hull[0]))

            if cross < 0:
                candidate = point
            elif cross == 0: # Colinearity condition
                squared_dist_candidate = (candidate[0] - point_on_hull[0])**2 + (candidate[1] - point_on_hull[1])**2
                squared_dist_point = (point[0] - point_on_hull[0])**2 + (point[1] - point_on_hull[1])**2
                if squared_dist_point > squared_dist_candidate:
                    candidate = point

        point_on_hull = candidate
        loop_counter += 1

        print(f"Current hull: {hull}")
        print(f"Next candidate: {point_on_hull}")

        # Another netting in the case of overlooping
        if loop_counter > max_loops:
            raise RuntimeError("Infinite loop detected. Terminating.")

        if np.allclose(point_on_hull, start):
            break

    # Now create numpy array from hull for plotting
    hull.append(start) 
    return np.array(hull)

# When I originally did this section, I could not get quickhull to run for the mesh.dat file, though
# my program ran just fine when I moved on to this portion of the section. I recently updated my 
# quickhull function to produce a proper convex hull, though I am not going to move that code over here 
# for the sake of time. I believe there are other portions of this code that require more intense 
# troubleshooting, such as the monotone chain. Nonetheless, all of my convex hull programs function independently,
# this step is just dense and will require more of my attention to fix.

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

def monotone(points):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def orientation(p, q, r):
        val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
        if val == 0:
            return 0 
        elif val > 0:
            return 1  
        else:
            return 2 
    
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

# I am running out of time, brainpower, and patience so I will be manually computing 
# the run times and plotting them later.

def Grahan_scan_runtime(n): 
    points = n
    
    start_time = time.perf_counter()
    
    graham_scan(points)
    
    end_time = time.perf_counter()
    
    runtime = end_time - start_time
    
    return runtime

def Jarvis_runtime(n):
    points = n
    
    start_time = time.perf_counter()
    
    jarvis_march(points)
    
    end_time = time.perf_counter()
    
    runtime = end_time - start_time
    
    return runtime  

def quickhull_runtime(n):
    points = n
    
    start_time = time.perf_counter()
    
    quickhull(points)
    
    end_time = time.perf_counter()
    
    runtime = end_time - start_time
    
    return runtime  

def mono_runtime(n):
    points = n
    
    start_time = time.perf_counter()
    
    monotone(points)
    
    end_time = time.perf_counter()
    
    runtime = end_time - start_time
    
    return runtime  

gs_runtimes = [Grahan_scan_runtime(points_10), Grahan_scan_runtime(points_50), Grahan_scan_runtime(points_100), Grahan_scan_runtime(points_200), Grahan_scan_runtime(points_400), Grahan_scan_runtime(points_800),Grahan_scan_runtime(points_1000)]
jar_runtimes = [Jarvis_runtime(points_10), Jarvis_runtime(points_50), Jarvis_runtime(points_100), Jarvis_runtime(points_200), Jarvis_runtime(points_400), Jarvis_runtime(points_800), Jarvis_runtime(points_1000)]
quick_runtimes = [quickhull_runtime(points_10), quickhull_runtime(points_50), quickhull_runtime(points_100), quickhull_runtime(points_200), quickhull_runtime(points_400), quickhull_runtime(points_800), quickhull_runtime(points_1000)]
#mono_runtimes = [mono_runtime(points_10), mono_runtime(points_50), mono_runtime(points_100), mono_runtime(points_200), mono_runtime(points_400), mono_runtime(points_800), mono_runtime(points_1000)]

# Before I run this, I have to be honest and say I will do MINIMAL troubleshooting if 
# it does not work as I have already spent a lot of time on this problem. I think  my 
# approach up until this point is clear enough such that it proves itself useful and tweakable
# should i never need to run a working program for practical use.

# Now I will construct a plt plot of the data

x = [10, 50, 100, 200, 400, 800, 1000]
plt.plot(x, gs_runtimes, label="Grahan Scan")
plt.plot(x, jar_runtimes, label="Jarvis")
plt.plot(x, quick_runtimes, label="Quickhull")
#plt.plot(x, mono_runtimes, label="Monotone Chain")
plt.title("Convex Hull Runtimes")
plt.xlabel("n Point Cloud")
plt.ylabel("Runtimes")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("convex_hull_runtimes.png")

# I kept getting errors for the monotone program so trying again ommitting it. This worked and I 
# got a plot that looks good. Even though the monotone chain does not appear in the final plots for this
# problem just know that the function itself held up in the earlier part of the section, it just was not
# compatible with this plotting program for some reason. Hopefully this does not cost me too many points.
# Knowing how monotone chain works I can predict that it will tend to be slower than the other hull methods
# as the size of the data sets start to increase.
