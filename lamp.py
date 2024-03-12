#%%
from ocp_vscode import *
from build123d import *
import math

n_horizontal_edges = 2
n_vertical_edges = 10
material_thickness=6
strap_width = 30
x_a = 100
y_a=100
x_b = 300
y_b = 100
tool_radius=3


def offset_y_coordinates(control_points, offset):
    return [(x, y + offset) for x, y in control_points]

controlpoints = [(0,0),(x_a,y_a), (200,100), (x_b, y_b), (400,0)]


def generate_interior_edge(controlpoints, offset):
    controlpoints_offset = offset_y_coordinates(controlpoints, -offset)
    s = Spline(*controlpoints_offset) 
    intersections = s.edges()[0].intersections(Axis.X)
    #remove part of the curve with negative y
    s = s - Rectangle(controlpoints_offset[-1][0] - controlpoints_offset[0][0],100,align=(Align.MIN, Align.MAX))
    return s, intersections


def generate_vertical_edges(controlpoints,n_vertical_edges):
    lamp_flat = []
    lamp = []
    radius = material_thickness/(2*math.sin(math.pi/n_vertical_edges))
    c = Pos(X=x_a)*Rot(Y=90)*extrude(Circle(y_a + radius - strap_width/2), material_thickness/2, both=True)
    c2 = Pos(X=x_b)*Rot(Y=90)*extrude(Circle(y_b + radius - strap_width/2), material_thickness/2, both=True)
    for i in range(n_vertical_edges):     
        temp = controlpoints.copy()
        temp[2] = (controlpoints[2][0], controlpoints[2][1]*(1+0.4*(1 - math.cos(i*math.radians(360)/n_vertical_edges +math.radians(180)))))
        exterior_spline = Spline(*temp)
        interior_spline, intersections = generate_interior_edge(temp, strap_width)
        connecting_line1 = Line(controlpoints[0], intersections[0])
        connecting_line2 = Line(controlpoints[-1], intersections[1])
        strap = [exterior_spline, connecting_line2, interior_spline, connecting_line1]
        a = Pos(Y=radius)*extrude(make_face(strap),material_thickness/2, both=True) -c -c2
        lamp_flat.append((Pos(Y=2*i*strap_width)*a))
        a = Rot(X=i*360/n_vertical_edges)*a
        lamp.append(a)
        

    m = Pos(X=x_a)*Rot(Y=90)*extrude(Circle(y_a + radius - strap_width/4) - Circle(y_a- strap_width), material_thickness/2, both=True) - lamp 
    n = Pos(X=x_b)*Rot(Y=90)*extrude(Circle(y_b + radius - strap_width/4) - Circle(y_b - strap_width), material_thickness/2, both=True) - lamp
    lamp_flat.append(Pos(110, 820)*Rot(Y=-90)*Pos(X=-x_a)*m)
    lamp_flat.append(Pos(330, 820)*Rot(Y=-90)*Pos(X=-x_b)*n)
    lamp.append(m)
    lamp.append(n)
    lamp = Compound(label="lamp", children=lamp)
    lamp_flat = Compound(label="lamp_flat", children=lamp_flat)
    return lamp, lamp_flat

        
lamp, lamp_flat = generate_vertical_edges(controlpoints, n_vertical_edges)

lamp.export_step("lamp.step")
lamp.export_stl("lamp.stl")
lamp_flat.export_step("lamp_flat.step")
lamp_flat.export_stl("lamp_flat.stl")
show(lamp)



# %%
