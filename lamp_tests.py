
# %%
from ocp_vscode import *
from build123d import *
import math


n_horizontal_edges = 2
n_vertical_edges = 10
material_thickness=0.12
strap_width = 0.3
pf_depth = 0.1
pf_width = 0.3
x_a = 1
y_a=1
x_b = 3
y_b = 1
controlpoints = [(0,0),(x_a,y_a), (2,1), (x_b, y_b), (4,0)]


def offset_y_coordinates(control_points, offset):
    return [(x, y + offset) for x, y in control_points]


def create_press_fit(edge, x_list, width, depth):
    pf = []
    new_curve = edge
    for x in x_list:
        p1 = edge.intersections(Axis(origin=(x - width/2,0, 0), direction=(0,1,0)))[0]
        p2 = edge.intersections(Axis(origin=(x + width/2 ,0, 0), direction=(0,1,0)))[0]
        if p1.Y < p2.Y:
            p1, p2 = p2, p1
        l1 = Line((p1.X, p1.Y), (p1.X, p1.Y-depth))
        l2 = Line((p1.X, p1.Y-depth), (p2.X, p1.Y-depth))
        l3 = Line((p2.X, p1.Y-depth), (p2.X, p2.Y))
        pf+=[l1, l2, l3]
        new_curve = new_curve - Pos(X=x)*Rectangle(width,1000)
    pf.append(new_curve)
    return pf



def generate_interior_edge(controlpoints, offset):
    controlpoints_offset = offset_y_coordinates(controlpoints, -offset)
    s = Spline(*controlpoints_offset) 
    intersections = s.edges()[0].intersections(Axis.X)
    #remove part of the curve with negative y
    s = s - Rectangle(controlpoints[-1][0] - controlpoints[0][0],1 ,align=(Align.MIN, Align.MAX))
    return s, intersections


def generate_vertical_edges(controlpoints,n_vertical_edges):
    lamp = []
    radius = material_thickness/(2*math.sin(math.pi/n_vertical_edges))
    for i in range(n_vertical_edges):     
        controlpoints[2] = (controlpoints[2][0], controlpoints[2][1]*(1+0.15*(1/2 - math.cos((i*math.radians(360))/n_vertical_edges)/2)))
        exterior_spline = Spline(*controlpoints)
        interior_spline, intersections = generate_interior_edge(controlpoints, strap_width)
        interior_spline = create_press_fit(interior_spline.edges()[0], [x_a, x_b], material_thickness, -pf_depth)
        connecting_line1 = Line(controlpoints[0], intersections[0])
        connecting_line2 = Line(controlpoints[-1], intersections[1])
        strap = [exterior_spline, connecting_line2, interior_spline, connecting_line1]
        a = Rot(X=i*360/n_vertical_edges)*Pos(Y=radius)*extrude(make_face(strap),material_thickness/2, both=True) 
        lamp.append(a)

    c = Pos(X=3)*Rot(Y=90)*extrude(Circle(y_a), material_thickness/2, both=True)
    c2 = Pos(X=1)*Rot(Y=90)*extrude(Circle(y_b), material_thickness/2, both=True)
    lamp.append(c)
    lamp.append(c2)
    return lamp

        
a = generate_vertical_edges(controlpoints, n_vertical_edges)
show(a)

#show(edges, c, c2)

# %%
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh


# Define sphere parameters
radius = 1  # Radius of the sphere

# Generate spherical coordinates with variations for a more organic shape
phi_varied = np.linspace(0, np.pi, 100)  # Polar angle
theta_varied = np.linspace(0, 2 * np.pi, 100)  # Azimuthal angle
phi_varied, theta_varied = np.meshgrid(phi_varied, theta_varied)

# Introducing variations to the radius to create an organic effect
# Adding a sinusoidal perturbation based on phi and theta to simulate bumps and dips
radius_varied = radius + 0.2 * np.sin(2*phi_varied) * np.cos(4 * theta_varied)

# Convert spherical coordinates to Cartesian coordinates for plotting the organic shape
x_varied = radius_varied * np.sin(phi_varied) * np.cos(theta_varied)
y_varied = radius_varied * np.sin(phi_varied) * np.sin(theta_varied)
z_varied = radius_varied * np.cos(phi_varied)


def surface_to_mesh(x, y, z, filename='surface_mesh.stl'):
    """
    Convert a surface defined by Cartesian coordinates (x, y, z) into a mesh and save as an STL file.

    Parameters:
    - x, y, z: 2D arrays of Cartesian coordinates defining the surface.
    - filename: String, the name of the STL file to save.
    """
    rows, cols = x.shape
    faces = []

    # Create faces (triangles) for the mesh
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Indices of the vertices for two triangles
            v0 = i * cols + j
            v1 = v0 + 1
            v2 = v0 + cols
            v3 = v2 + 1

            # Append triangles by vertex indices
            faces.append([v0, v1, v3])
            faces.append([v0, v3, v2])

    # Flatten the arrays for easy indexing
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    
    # Create the mesh
    surface_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            surface_mesh.vectors[i][j] = points[f[j], :]

    # Save the mesh to file
    surface_mesh.save(filename)

# Given Cartesian coordinates (x_varied, y_varied, z_varied) from the surface creation code
surface_to_mesh(x_varied, y_varied, z_varied, 'organic_surface.stl')


# Plotting the organic shape
fig_varied = plt.figure(figsize=(8, 6))
ax_varied = fig_varied.add_subplot(111, projection='3d')
ax_varied.plot_surface(x_varied, y_varied, z_varied, color='c', edgecolor='k', alpha=0.6)
ax_varied.set_title('3D Organic Closed Hypersurface for a Lamp')
plt.tight_layout()
plt.show()



# %%
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh


# Define sphere parameters
radius = 1  # Radius of the sphere

# Generate spherical coordinates with variations for a more organic shape
phi_varied = np.linspace(0, np.pi, 100)  # Polar angle
theta_varied = np.linspace(0, 2 * np.pi, 100)  # Azimuthal angle
phi_varied, theta_varied = np.meshgrid(phi_varied, theta_varied)

# Introducing variations to the radius to create an organic effect
# Adding a sinusoidal perturbation based on phi and theta to simulate bumps and dips
radius_varied = radius + 0.2 * np.sin(2*phi_varied) * np.cos(4 * theta_varied)

# Convert spherical coordinates to Cartesian coordinates for plotting the organic shape
x_varied = radius_varied * np.sin(phi_varied) * np.cos(theta_varied)
y_varied = radius_varied * np.sin(phi_varied) * np.sin(theta_varied)
z_varied = radius_varied * np.cos(phi_varied)