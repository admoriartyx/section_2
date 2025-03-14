# Part a

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def surface1(x, y):
    return 2 * x**2 + 2 * y**2

def surface2(x, y):
    return 2 * np.exp(-x**2 - y**2)

def generate_surface_point_cloud(num_points=100):
    x = np.linspace(-1, 1, num_points)
    y = np.linspace(-1, 1, num_points)
    X, Y = np.meshgrid(x, y)
    Z1 = surface1(X, Y)
    Z2 = surface2(X, Y)
    
    return X, Y, Z1, Z2

X, Y, Z1, Z2 = generate_surface_point_cloud()

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
z_difference = np.abs(Z1 - Z2)
colors1 = np.zeros((Z1.shape[0], Z1.shape[1], 4))
colors2 = np.zeros((Z2.shape[0], Z2.shape[1], 4))

for i in range(Z1.shape[0]):
    for j in range(Z1.shape[1]):
        if z_difference[i, j] < 0.5:
            colors1[i, j] = [1, 0.5, 0, 0.8] 
            colors2[i, j] = [1, 0.5, 0, 0.8]
        else:
            colors1[i, j] = [0, 0, 1, 0.2] 
            colors2[i, j] = [1, 0, 0, 0.2] 

ax.plot_surface(X, Y, Z1, facecolors=colors1)
ax.plot_surface(X, Y, Z2, facecolors=colors2)

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.view_init(elev=25, azim=-45)

plt.show()
plt.savefig('T2_part_a.png')

# Part b

from scipy.spatial import Delaunay
from scipy.optimize import fsolve

def find_boundary_radius():
    func = lambda r: r**2 - np.exp(-r**2)
    r_boundary = fsolve(func, 0.7)[0]
    return r_boundary

def generate_grid_points(R, num_points=50):
    x = np.linspace(-R, R, num_points)
    y = np.linspace(-R, R, num_points)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    mask = xv**2 + yv**2 <= R**2
    xv = xv[mask]
    yv = yv[mask]
    points = np.vstack([xv, yv]).T
    return points

def generate_surface_point_cloud(points):
    z_top = surface2(points[:, 0], points[:, 1])
    z_bottom = surface1(points[:, 0], points[:, 1])
    top_points = np.hstack([points, z_top[:, None]])
    bottom_points = np.hstack([points, z_bottom[:, None]])
    return top_points, bottom_points

def is_boundary_point(point, R, tol=1e-3):
    r = np.sqrt(point[0]**2 + point[1]**2)
    return np.isclose(r, R, atol=tol)

def main():
    R = find_boundary_radius()
    print("Boundary radius R =", R)
    grid_points = generate_grid_points(R, num_points=50)
    top_points, bottom_points = generate_surface_point_cloud(grid_points)
    tri_top = Delaunay(grid_points)
    tri_bottom = Delaunay(grid_points)
    boundary_indices = []
    for i, pt in enumerate(grid_points):
        if is_boundary_point(pt, R, tol=1e-3):
            boundary_indices.append(i)
    boundary_indices = np.array(boundary_indices)
    print("Found", len(boundary_indices), "boundary points.")
    
    num_top = len(top_points)
    bottom_mapping = {}
    bottom_interior_points = []
    for i, pt in enumerate(bottom_points):
        if is_boundary_point(grid_points[i], R, tol=1e-3):
            bottom_mapping[i] = i 
        else:
            bottom_mapping[i] = num_top + len(bottom_interior_points)
            bottom_interior_points.append(pt)
    bottom_interior_points = np.array(bottom_interior_points)
    combined_vertices = np.vstack([top_points, bottom_interior_points])
    bottom_triangles = []
    for tri in tri_bottom.simplices:
        new_tri = [bottom_mapping[idx] for idx in tri]
        bottom_triangles.append(new_tri)
    bottom_triangles = np.array(bottom_triangles)
    
    top_triangles = tri_top.simplices
    combined_triangles = np.vstack([top_triangles, bottom_triangles])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(
        combined_vertices[:, 0],
        combined_vertices[:, 1],
        combined_vertices[:, 2],
        triangles=combined_triangles,
        cmap='viridis',
        edgecolor='none',
        alpha=0.8
    )
    ax.set_title("Closed Surface Triangulation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
    plt.savefig('T2_part_b.png')

if __name__ == '__main__':
    main()

# Parts c and d essentially ask for the same thing so my next set of code will encompass both parts.


from collections import Counter
from matplotlib import cm

def surface1(x, y):
	return 2*x**2 + 2*y**2

def surface2(x, y):
	return 2*np.exp(-x**2 - y**2)

x = np.linspace(-1, 1, 51)
y = np.linspace(-1, 1, 51)
z = np.linspace(0, 2, 21)
x, y, z = np.meshgrid(x, y, z)
indices = np.logical_and(surface1(x, y) <= z, z <= surface2(x, y))
x = x[indices]
y = y[indices]
z = z[indices]
tetrahedra = Delaunay(np.c_[x, y, z]).simplices
face_counter = Counter()
for vertices in tetrahedra:
	i1, i2, i3, i4 = sorted(vertices)
	face_counter.update([(i1, i2, i3), (i1, i2, i4), (i1, i3, i4), (i2, i3, i4)])
boundary_faces = [face for face, count in face_counter.items() if count == 1]

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_trisurf(x, y, z, triangles=boundary_faces, cmap=cm.viridis)
plt.show()
plt.savefig('T2_part_c&d.png')

# The outputs of parts b and d differ in thr degree of triangulation for the mesh surface.
# The triangles appear to be a lot smaller and finer for part b, whereas the part d output sees 
# far more distinct triangles on the surface.

