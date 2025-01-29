# Starting off with Task 1
# 1. Build Algorithms
# Part a:
import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt('mesh.dat', skip_header=1)
x, y = data[:, 0], data[:, 1]

plt.scatter(x, y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Point Cloud Visualization")
plt.show()
plt.savefig('Point_Cloud.png')

# Parts b and c of task 1 are completed in invidual .py programs for each of the convex hull 
# methods. 


