# Now moving on to part b of task 1
# This code is specifically for the Graham Scan.

import numpy as np
import matplotlib.pyplot as plt

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

# In the case of our mesh.dat file, we need the program to accept points via reading a data file
def file_reader(filename):
    try:
        data = np.loadtxt(filename, skiprows=1)
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# Main program
if __name__ == "__main__": 
    filename = 'mesh.dat' 

    # Data extraction
    points = file_reader(filename)
    if points is not None:
        hull = graham_scan(points)

        plt.scatter(points[:, 0], points[:, 1])  
        plt.plot(hull[:, 0], hull[:, 1], 'r-') 
        plt.scatter(hull[:, 0], hull[:, 1], color='red') 
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Convex Hull using Graham Scan")
        plt.grid(True)
        plt.show()
        plt.savefig('Graham_Scan_fig.png')
    else:
        print("Error")

