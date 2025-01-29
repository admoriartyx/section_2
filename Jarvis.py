# Now moving onto the Jarvis march method
import numpy as np
import matplotlib.pyplot as plt

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

def read_points_from_file(filename):
    try:
        # Skipping header b/c mesh.dat has x,y column heads
        data = np.loadtxt(filename, skiprows=1)
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

if __name__ == "__main__":
    # This section of the code is meant to extract points from a data file,
    # as opposed to assigning specific points manually
    filename = 'mesh.dat' 

    points = read_points_from_file(filename)
    if points is not None:

        try:
            # Program execution
            hull = jarvis_march(points)
            # Plotting!
            plt.scatter(points[:, 0], points[:, 1], label="Points")  
            plt.plot(hull[:, 0], hull[:, 1], 'r-')  
            plt.scatter(hull[:, 0], hull[:, 1], color='red')  
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("Convex Hull using Jarvis March")
            plt.grid(True)
            plt.show()
            plt.savefig('Jarvis_mesh_updated.png') # Had to add this b/c output wouldnt appear in my terminal
        except RuntimeError as e:
            print(e)
    else:
        print("Failed to read points from the file.")

