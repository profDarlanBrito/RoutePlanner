import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from numpy import dtype, floating, ndarray
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as Rot
from spherical_geometry.polygon import SphericalPolygon
from sympy import Eq, solve, symbols
from sympy.geometry import Plane, Point3D


def plot_circle(radius: float, resolution: int, ax=None):
    # Generate points along the circumference of the circle
    theta = np.linspace(0, 2 * np.pi, resolution)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(x)

    # Plot the circle in 3D
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, color="blue", linewidth=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Circle in 3D")
    return ax


def plot_circle_in_plane(normal, center, radius, ax=None):
    # Unpack the normal vector and center point
    a, b, c = normal
    x0, y0, z0 = center

    # Generate a unit vector orthogonal to the normal vector
    u = np.array([1, 0, 0])
    if np.dot(u, normal) == 1:
        u = np.array([0, 1, 0])
    u = u - np.dot(u, normal.T) * normal
    u = u / np.linalg.norm(u)

    # Generate another unit vector orthogonal to the normal vector and u
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    # Generate points along the circumference of the circle
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_points = np.array(
        [
            x0 + radius * (np.cos(theta) * u[0] + np.sin(theta) * v[0]),
            y0 + radius * (np.cos(theta) * u[1] + np.sin(theta) * v[1]),
            z0 + radius * (np.cos(theta) * u[2] + np.sin(theta) * v[2]),
        ]
    )

    if ax is None:
        # Plot the circle in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(circle_points[0], circle_points[1], circle_points[2], color="blue", linewidth=2)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Circle in Plane with Normal Vector")
    else:
        ax.plot(circle_points[0], circle_points[1], circle_points[2], color="blue", linewidth=2)
    return ax


def plot_line(vector, point, ax):
    # Unpack the vector and point
    a, b, c = vector
    x0, y0, z0 = point

    # Define the range of t
    t = np.linspace(-1, 1, 100)

    # Calculate points on the line
    x = x0 + a * t
    y = y0 + b * t
    z = z0 + c * t

    if ax is None:
        # Plot the line in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x0, y0, z0, color="red", marker="-o", label="Point (x0, y0, z0)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Line in 3D")
        ax.legend()
    else:
        ax.plot(x, y, z, color="blue", linewidth=2)
    return ax


def plot_plane_through_points(point1, point2, point3, ax=None):
    # Define the points
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    x3, y3, z3 = point3

    # Calculate the normal vector
    v1 = np.array([x2 - x1, y2 - y1, z2 - z1])
    v2 = np.array([x3 - x1, y3 - y1, z3 - z1])
    normal = np.cross(v1, v2)

    # Define a grid of points to plot the plane
    xx, yy = np.meshgrid(np.linspace(x1 - 1, x3 + 1, 10), np.linspace(y1 - 1, y3 + 1, 10))
    zz = (-normal[0] * (xx - x1) - normal[1] * (yy - y1)) / normal[2] + z1

    if ax is None:
        # Plot the plane in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(xx, yy, zz, alpha=0.5)

        # Plot the points
        ax.scatter(x1, y1, z1, color="red", marker="o", label="Point 1")
        ax.scatter(x2, y2, z2, color="green", marker="o", label="Point 2")
        ax.scatter(x3, y3, z3, color="blue", marker="o", label="Point 3")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Plane through Three Points")
        ax.legend()
    else:
        ax.scatter(x1, y1, z1, color="red", marker="o", label="Point 1")
        ax.scatter(x2, y2, z2, color="green", marker="o", label="Point 2")
        ax.scatter(x3, y3, z3, color="blue", marker="o", label="Point 3")
    return normal, ax


def plot_hemisphere(center, radius, theta, ax=None):
    # Unpack the center
    a, b, c = center

    # Generate points on the hemisphere surface
    phi = np.linspace(0, np.pi, 50)
    theta_rad = np.deg2rad(theta)
    # x = a + radius * np.outer(np.sin(phi), np.cos(theta_rad))
    # y = b + radius * np.outer(np.sin(phi), np.sin(theta_rad))
    # z = c + radius * np.outer(np.cos(phi), np.ones_like(theta_rad))

    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi / 2, 50)
    x = a + radius * np.outer(np.cos(u), np.sin(v))
    y = b + radius * np.outer(np.sin(u), np.sin(v))
    z = c + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    if ax is None:
        # Plot the hemisphere in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, z, color="blue", alpha=0.8)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Hemisphere")
    else:
        ax.plot_surface(x, y, z, color="blue", alpha=0.8)
    return ax, x, y, z


def plot_plane(plotter, normal_vec, point_plane, color="red", opacity=0.2):
    plane = pv.Plane(point_plane, normal_vec, i_size=9, j_size=9)
    plotter.add_mesh(plane, color=color, opacity=opacity)


def compute_orientation(normal):
    # Normalize plane normal
    normal = normal / np.linalg.norm(normal)

    # Compute the quaternion orientation
    # From the plane normal
    rot = Rotation.from_rotvec(np.pi / 2 * normal)
    return rot


def get_geometric_objects_cell(geometric_objects):
    for i in range(geometric_objects.n_cells):
        yield geometric_objects.get_cell(i)


def find_normal_vector(point1, point2, point3):
    vec1 = np.array(point2) - np.array(point1)
    vec2 = np.array(point3) - np.array(point1)
    cross_vec = np.cross(vec1, vec2)
    return cross_vec / np.linalg.norm(cross_vec)


def compute_area_normal_hemisphere(c0: ndarray, p0: ndarray, n: ndarray, s_normal: ndarray, R: float) -> float:
    """
    Args:
        c0 (ndarray): center of sphere
        p0 (ndarray): arbitrary point on plane
        n (ndarray): normal vector of plane
        s_normal (ndarray): normal vector of sphere
        R (float): radius of sphere

    Returns:
        float: area of sphere
    """

    if not np.isclose(s_normal @ n / np.linalg.norm(s_normal) / np.linalg.norm(n), -1):
        return 0.0

    rho: float = (c0 - p0) @ n / np.linalg.norm(n)
    r_2: float = R**2 - rho**2

    if rho >= 0:
        return 2 * np.pi * np.sqrt(r_2) * (R - rho) if r_2 > 0 else 0.0
    else:
        return 2 * np.pi * R**2


def get_plane_frustum(frustum: pv.PolyData) -> list[tuple[ndarray, ndarray]]:
    # above_plane
    # bellow_plane
    # right_plane
    # left_plane
    # far_plane
    # near_plane
    planes = []

    for cell in get_geometric_objects_cell(frustum):
        pos_cell = np.array(cell.center)
        points_cell = cell.points[:3]
        norm_vec = find_normal_vector(*points_cell)
        planes.append((norm_vec, pos_cell))

    return planes


def get_close_intersection_points(intersection, camera_position, value):
    try:
        x0 = float(intersection[0][0].subs(z, value))
        y0 = float(intersection[0][1].subs(z, value))
        z0 = float(intersection[0][2].subs(z, value))

        x1 = float(intersection[1][0].subs(z, value))
        y1 = float(intersection[1][1].subs(z, value))
        z1 = float(intersection[1][2].subs(z, value))
    except TypeError:
        return np.array([0, 0, 0])

    d0 = np.linalg.norm(np.array([x0, y0, z0]) - np.array(camera_position))
    d1 = np.linalg.norm(np.array([x1, y1, z1]) - np.array(camera_position))

    return np.array([x0, y0, z0]) if d0 < d1 else np.array([x1, y1, z1])


def get_point_intersection_plane_with_sphere(
    pi_sphere: ndarray, pi_frustum: ndarray, position_sphere: ndarray, camera_position: ndarray, radius_sphere: float
) -> list[ndarray]:
    """
    Args:
        pi_sphere (ndarray): _description_
        pi_frustum (ndarray): _description_
        position_sphere (ndarray): _description_
        radius_sphere (float): _description_

    Returns:
        _type_: _description_
    """
    print("get_point_intersection_plane_with_sphere")
    parametric_equation = get_line_of_intersection_two_planes(pi_sphere, pi_frustum)

    intersection_points = []
    points = get_intersection_points_line_sphere(parametric_equation, (*position_sphere, radius_sphere))
    intersection_points = [p for p in points]

    intersection1 = plane_with_circle_intersection(pi_sphere, [*position_sphere, radius_sphere])
    intersection2 = plane_with_circle_intersection(pi_frustum, [*position_sphere, radius_sphere])

    dz = position_sphere[2]
    intersection_points.append(get_close_intersection_points(intersection1, camera_position, dz))
    intersection_points.append(get_close_intersection_points(intersection2, camera_position, dz))

    return intersection_points


def great_circle_distance(point1, point2, radius):
    # Convert Cartesian coordinates to spherical coordinates
    x1, y1, z1 = point1
    x2, y2, z2 = point2

    # Calculate angular distance between the points
    dot_product = x1 * x2 + y1 * y2 + z1 * z2
    magnitude_product = math.sqrt(x1**2 + y1**2 + z1**2) * math.sqrt(x2**2 + y2**2 + z2**2)
    angle = math.acos(dot_product / magnitude_product)

    # Calculate arc length (distance) on the sphere
    distance = radius * angle
    return distance


def triangle_area(point1: ndarray, point2: ndarray, point3: ndarray, radius_ta: float):
    d1 = great_circle_distance(point1, point2, radius_ta)
    d2 = great_circle_distance(point1, point3, radius_ta)
    d3 = great_circle_distance(point1, point3, radius_ta)

    # Calculate the semi-perimeter
    s = (d1 + d2 + d3) / 2
    # Calculate the area using Heron's formula
    area = (s * (s - d1) * (s - d2) * (s - d3)) ** 0.5
    return area


def calculate_spherical_side_area(
    sphere_mesh: pv.PolyData, intersection_points: list[ndarray], plane_eq: list, radius_cssa: float = None
) -> float:
    """
    Calculates the area of the portion of a sphere that lies on one side of a plane.

    Args:
        sphere_mesh (Any): The mesh representation of the sphere.
        intersection_points (list[np.ndarray]): List of intersection points between the sphere and the plane.
        plane_eq: Equation of the plane in the form (a, b, c, d) where ax + by + cz + d = 0.

    Returns:
        float: The area of the spherical side.
        :param radius_cssa:
    """
    sign = np.sign(np.array(plane_eq[:3] @ np.array(intersection_points[-1])) + plane_eq[-1])

    sphere_area = 0
    for cell in get_geometric_objects_cell(sphere_mesh):
        points = cell.points

        count = 0
        for p in points:
            p_sign = np.sign(np.array(plane_eq[:3] @ np.array(p)) + plane_eq[-1])

            if p_sign == sign:
                count += 1

        if count < 2:
            continue

        sphere_area += triangle_area(points[0], points[1], points[2], radius_cssa)

    return sphere_area


def get_viewed_area():
    # Create some sample meshes
    pos_mesh = np.array([0, 0, 0])
    r_mesh = 2

    cam_pos = (2.9, -1.05, 0.0)
    sphe_direction = np.array(cam_pos) - pos_mesh
    mesh = pv.Sphere(radius=r_mesh, center=pos_mesh, direction=sphe_direction, phi_resolution=10, end_phi=90)
    mesh1 = pv.Box(bounds=(-5.0, -4.0, -1.0, 1.0, -1.0, 1.0))

    # Create a plotter
    plotter = pv.Plotter()

    cy_direction = np.array([0, 0, 1])
    cy_hight = 0.4
    n_resolution = 36

    # Calculate the length of the lateral surface of an inscribed cylinder
    h = np.cos(np.pi / n_resolution) * r_mesh
    l = np.sqrt(np.abs(4 * h**2 - 4 * r_mesh**2))

    # Find the radius of the spheres
    z_resolution = int(np.ceil(cy_hight / l))
    h = cy_hight / z_resolution
    spheres_radius = np.max([l, h]) / 2

    cylinder = pv.CylinderStructured(
        center=pos_mesh,
        direction=cy_direction,
        radius=r_mesh,
        height=cy_hight,
        theta_resolution=n_resolution,
        z_resolution=z_resolution,
    )

    # cylinder.plot(show_edges=True)
    # Add the meshes to the plotter
    # plotter.add_mesh(mesh1)
    # plotter.add_mesh(mesh, show_edges=True)
    plotter.add_mesh(cylinder, show_edges=True)

    # Set camera position and orientation (optional)
    plotter.camera.clipping_range = (1e-4, 1)
    plotter.camera_position = [cam_pos, (0, 0, 0), (0, 0, 0)]

    points = np.array([[2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [2.0, 0.0, 2.0], [2.0, 2.0, 2.0]])
    point_cloud = pv.PolyData(points)
    # plotter.add_mesh(point_cloud)

    # Get the camera's view frustum
    frustum = plotter.camera.view_frustum()
    plotter.add_mesh(frustum, style="wireframe")

    # Generate a plane
    direction = np.array(plotter.camera.focal_point) - np.array(plotter.camera.position)
    direction /= np.linalg.norm(direction)

    u = np.linalg.norm(pos_mesh - cam_pos) - r_mesh + 0.11
    v = pos_mesh - np.array(cam_pos)
    v /= np.linalg.norm(v)
    dot_plane = np.array([cam_pos[0] + u * v[0], cam_pos[1] + u * v[1], cam_pos[1] + u * v[2]])

    plane_frustum = get_plane_frustum(frustum)
    plane_eq = []

    # Plot the four sides of the frustum
    for plane in plane_frustum[:4]:
        a, b, c = plane[0]
        d = -(plane[0] @ plane[1])
        plane_eq.append([a, b, c, d])

    # plot_plane(plotter, *plane_frustum[2])
    # Calculate the intersection between the planes of the frustum
    parametric_equation = []
    # above_plane with left_plane
    parametric_equation.append(get_line_of_intersection_two_planes(plane_eq[0], plane_eq[2]))
    # above_plane with right_plane
    parametric_equation.append(get_line_of_intersection_two_planes(plane_eq[0], plane_eq[3]))
    # bellow_plane with left_plane
    parametric_equation.append(get_line_of_intersection_two_planes(plane_eq[1], plane_eq[2]))
    # bellow_plane with right_plane
    parametric_equation.append(get_line_of_intersection_two_planes(plane_eq[1], plane_eq[3]))

    # Select the points that pass through the sphere closest to the camera
    intersection_points = []
    for p_eq in parametric_equation:
        intersection_points_sphere = get_intersection_points_line_sphere(p_eq, (*pos_mesh, r_mesh))

        d1 = np.linalg.norm(intersection_points_sphere[0] - cam_pos)
        d2 = np.linalg.norm(intersection_points_sphere[1] - cam_pos)

        p = intersection_points_sphere[0] if d1 < d2 else intersection_points_sphere[1]
        # plotter.add_points(p, color="green", point_size=10)
        intersection_points.append(p)

    spherical_points = np.row_stack(intersection_points)
    spherical_polygon = SphericalPolygon(spherical_points)
    spherical_area = spherical_polygon.area()
    # print(f"{spherical_area=}")

    # plot_plane(plotter, direction, dot_plane)

    #### delete
    dist: float = np.inf
    best_plane = None
    best_pos = None
    best_mesh = None
    ####
    # Create the hemispheres and add them to the faces of the cylinder
    for cell in get_geometric_objects_cell(cylinder):
        pos_cell = cell.center
        points_cell = cell.points[:3]
        norm_vec = find_normal_vector(*points_cell)

        sub_mesh = pv.Sphere(radius=spheres_radius, center=pos_cell, direction=norm_vec, phi_resolution=10, end_phi=90)
        # plotter.add_mesh(sub_mesh, show_edges=True)

        # area_spheres += compute_area_normal_hemisphere(pos_cell, dot_plane, direction, norm_vec, spheres_radius)

        #### delete
        d = np.linalg.norm(np.array(pos_cell) - np.array(cam_pos))
        if d < dist:
            dist = d
            best_pos = pos_cell
            a, b, c = norm_vec
            d = -(norm_vec @ np.array(pos_cell))
            best_plane = [a, b, c, d]
            best_mesh = sub_mesh
        ####

    plotter.add_mesh(best_mesh, show_edges=True)
    intersection_points = get_point_intersection_plane_with_sphere(
        best_plane, plane_eq[3], best_pos, cam_pos, spheres_radius
    )
    print(f"{spheres_radius=}")
    plotter.add_points(intersection_points[-1], color="green", point_size=10)

    spherical_area = calculate_spherical_side_area(best_mesh, intersection_points, plane_eq[3], None)
    print(f"{spherical_area=}")

    # area_spheres = compute_area_normal_hemisphere(pos_mesh, dot_plane, direction, sphe_direction, r_mesh)

    bounds_mesh = mesh.bounds
    # Get the bounds of the meshes
    # bounds_mesh1 = mesh1.bounds

    # Calculate the intersection of the camera frustum and mesh bounds to find the viewed area
    # viewed_area_mesh = [max(bounds_mesh[0], frustum.bounds[0]), min(bounds_mesh[1], frustum.bounds[1]),
    #                      max(bounds_mesh[2], frustum.bounds[2]), min(bounds_mesh[3], frustum.bounds[3]),
    #                      max(bounds_mesh[4], frustum.bounds[4]), min(bounds_mesh[5], frustum.bounds[5])]
    bellow_plane = frustum.get_cell(0)
    above_plane = frustum.get_cell(1)
    right_plane = frustum.get_cell(2)
    left_plane = frustum.get_cell(3)  # Get a plane of the frustum
    far_clip = frustum.get_cell(4)
    near_clip = frustum.get_cell(5)
    points1 = np.empty((4, 3))
    c = 0
    for i in range(3):
        if i == 1:
            continue
        # Get each line on the border of the plane of the frustum
        line = above_plane.get_edge(i)
        # line.bounds get 6 numbers of the line [x_start,x_end,y_start,y_end,z_start,z_end]
        bounds_line = np.array(line.bounds)
        c_odd = 0
        c_even = 0
        for j in range(6):
            if j % 2:
                points1[c * 2, c_even] = bounds_line[j]
                c_even += 1
            else:
                points1[(c * 2) + 1, c_odd] = bounds_line[j]
                c_odd += 1
        c += 1

    # mesh = pv.Plane(center=point1, direction=normal, i_size=15, j_size=15)
    # Create a plane from the points (not working yet) Works with variable points but does not work with points1
    mesh_plane = create_mesh_from_points(points1)
    plotter.add_mesh(mesh_plane)  # Add the mesh of the plane to plotter figure
    plotter.show_grid()
    # Show the plotter
    plotter.show()

    if far_clip.bounds[0] < mesh.bounds[1] < near_clip.bounds[0]:
        print("Sphere viewed")

    viewed_area_mesh = np.mean(
        np.array(
            [
                (bounds_mesh[0] - frustum.bounds[0]),
                (bounds_mesh[1] - frustum.bounds[1]),
                (bounds_mesh[2] - frustum.bounds[2]),
                (bounds_mesh[3] - frustum.bounds[3]),
                (bounds_mesh[4] - frustum.bounds[4]),
                (bounds_mesh[5] - frustum.bounds[5]),
            ]
        )
    )

    # print("Viewed area of mesh1:", viewed_area_mesh1)
    print("Viewed area of mesh2:", viewed_area_mesh)


def create_mesh_from_points(points):
    # Create a mesh from four points
    mesh = pv.PolyData()
    mesh.points = np.array(points[0])  # point[]
    mesh.faces = np.array([4, 0, 1, 2, 3], np.int8)
    return mesh


# Define variables
x, y, z, t = symbols("x y z t")


def get_line_of_intersection_two_planes(pi1, pi2):
    # Define the equations of the planes
    plane1 = Eq(pi1[0] * x + pi1[1] * y + pi1[2] * z, -pi1[3])
    plane2 = Eq(pi2[0] * x + pi2[1] * y + pi2[2] * z, -pi2[3])

    # Solve the system of equations to find the direction vector
    direction_vector = np.cross(pi1[:3], pi2[:3])

    # Find a point on the line of intersection (by setting one variable to zero)
    # Here we set z = 0, you can choose any other variable as well
    point = solve((plane1, plane2))
    point[x] = point[x].subs(z, 0)
    point[y] = point[y].subs(z, 0)

    # Formulate the parametric equation of the line
    parametric_equation = [
        point[x] + direction_vector[0] * t,
        point[y] + direction_vector[1] * t,
        direction_vector[2] * t,
    ]
    return parametric_equation


def get_intersection_points_line_sphere(line_parametric_eq, sphere_eq):
    # Extract components of the line's parametric equations
    x_expr, y_expr, z_expr = line_parametric_eq

    # Extract components of the sphere equation
    x_sphere, y_sphere, z_sphere, r = sphere_eq

    # Substitute the parametric equations of the line into the equation of the sphere
    sphere_eq_subs = Eq((x_expr - x_sphere) ** 2 + (y_expr - y_sphere) ** 2 + (z_expr - z_sphere) ** 2, r**2)

    # Solve for t to find the point(s) of intersection
    solutions = solve(sphere_eq_subs, t)

    # Evaluate the parametric equations at the intersection point(s)
    intersection_points = np.empty([0, 3])

    try:
        for sol in solutions:
            x_inter = x_expr.subs(t, sol)
            y_inter = y_expr.subs(t, sol)
            z_inter = z_expr.subs(t, sol)
            intersection_points = np.row_stack((intersection_points, (float(x_inter), float(y_inter), float(z_inter))))
    except TypeError:
        return np.array([])

    return intersection_points


def get_line_of_intersection_two_planes_no_sym(pi1, pi2):
    # Extract coefficients of the planes
    a1, b1, c1, d1 = pi1
    a2, b2, c2, d2 = pi2

    # Define the direction vector of the line of intersection
    direction_vector = np.array([b1 * c2 - b2 * c1, a2 * c1 - a1 * c2, a1 * b2 - a2 * b1])

    # Find a point on the line of intersection (by setting one variable to zero)
    # Here we set z = 0, you can choose any other variable as well
    point = np.array([0.0, 0.0, 0.0])
    if direction_vector[0] != 0:
        point[0] = 0
        point[1] = (a1 * c2 * d2 - a2 * c1 * d1) / (a1 * c2 - a2 * c1)
        point[2] = (b1 * c2 * d2 - b2 * c1 * d1) / (b1 * c2 - b2 * c1)
    elif direction_vector[1] != 0:
        point[0] = (b1 * d2 - b2 * d1) / (b1 * c2 - b2 * c1)
        point[1] = 0
        point[2] = (a2 * d1 - a1 * d2) / (a1 * c2 - a2 * c1)
    else:
        point[0] = (b1 * d2 - b2 * d1) / (b1 * a2 - b2 * a1)
        point[1] = (a2 * d1 - a1 * d2) / (a1 * b2 - a2 * b1)
        point[2] = 0

    # Formulate the parametric equation of the line
    def line_equation(t):
        return point + direction_vector * t

    return line_equation


def get_intersection_points_line_sphere_no_sym(line_points, sphere_eq):
    # Extract components of the line's points
    x1, y1, z1 = line_points[0]
    x2, y2, z2 = line_points[1]

    # Extract components of the sphere equation
    x_sphere, y_sphere, z_sphere, r = sphere_eq

    # Calculate the direction vector of the line
    direction_vector = np.array([x2 - x1, y2 - y1, z2 - z1])

    # Calculate the vector from one of the line's points to the center of the sphere
    vector_to_center = np.array([x_sphere - x1, y_sphere - y1, z_sphere - z1])

    # Calculate the dot product of the direction vector and the vector to the center
    dot_product = np.dot(direction_vector, vector_to_center)

    # Calculate the discriminant
    discriminant = dot_product**2 - np.dot(direction_vector, direction_vector) * (
        np.dot(vector_to_center, vector_to_center) - r**2
    )

    # Check if the discriminant is negative, indicating no intersection
    if discriminant < 0:
        return np.array([])

    # Calculate the parameter values for the intersection points
    t1 = (-dot_product + np.sqrt(discriminant)) / np.dot(direction_vector, direction_vector)
    t2 = (-dot_product - np.sqrt(discriminant)) / np.dot(direction_vector, direction_vector)

    # Calculate the intersection points
    intersection_point1 = np.array([x1 + t1 * (x2 - x1), y1 + t1 * (y2 - y1), z1 + t1 * (z2 - z1)])
    intersection_point2 = np.array([x1 + t2 * (x2 - x1), y1 + t2 * (y2 - y1), z1 + t2 * (z2 - z1)])

    return np.array([intersection_point1, intersection_point2])


def spherical_distance(p1, p2):
    """
    Calculate the spherical distance between two points on a unit sphere.
    """

    # Convert spherical coordinates to radians
    lon1, lat1 = math.radians(p1[0]), math.radians(p1[1])
    lon2, lat2 = math.radians(p2[0]), math.radians(p2[1])

    # Calculate spherical distance using haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = c

    return distance


def spherical_triangle_area(p1, p2, p3):
    """
    Calculate the area of a spherical triangle formed by three points on a unit sphere.
    """
    # Calculate the lengths of the three sides of the spherical triangle
    side1 = spherical_distance(p1, p2)
    side2 = spherical_distance(p2, p3)
    side3 = spherical_distance(p3, p1)

    # Calculate the semi-perimeter
    s = (side1 + side2 + side3) / 2

    # Calculate the spherical excess using Heron's formula
    excess = 4 * math.atan(
        math.sqrt(math.tan(s / 2) * math.tan((s - side1) / 2) * math.tan((s - side2) / 2) * math.tan((s - side3) / 2))
    )

    # The area of the spherical triangle is equal to its excess angle
    area = excess

    return area


def plane_circle_intersection(plane_eq, circle):
    # Extract components of the plane equation
    a, b, c, d = plane_eq

    # Define the plane
    plane = Plane(Point3D(0, 0, -d / c), normal_vector=(a, b, c))

    # Project the circle onto the plane
    projected_circle = circle.projection(plane)

    # Find the intersection points between the projected circle and the plane
    intersection_points_ci = projected_circle.intersection(plane)

    return intersection_points_ci


def plane_with_circle_intersection(plane_eq, circle_eq):
    """
    Args:
        plane_eq (array_like): ax + by + cy + d = 0
        circle_eq (array_like): (x - cx)^2 + (y - cy)^2 + (z - cz)^2 = r^2

    """
    # Extract components of the plane equation
    a, b, c, d = plane_eq
    cx, cy, cz, r = circle_eq

    plane = a * x + b * y + c * z + d
    sphere = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 - r**2

    solution = solve((plane, sphere), (x, y, z))

    return solution


def get_viewed_area_from():
    print("Starting viewed area computing")


def draw_cylinder_with_hemisphere(
    plotter,
    cy_direction: ndarray,
    cy_height: float,
    n_resolution: int,
    cy_radius: float,
    cy_center: ndarray,
    low_cylinder_limit=0.0,
):
    print("Drawing cylinder with hemispheres")
    meshes = {}
    # Calculate the length of the lateral surface of an inscribed cylinder
    h = np.cos(np.pi / n_resolution) * cy_radius
    l = np.sqrt(np.abs(4 * h**2 - 4 * cy_radius**2))

    # Find the radius of the spheres
    z_resolution = int(np.ceil(cy_height / l))
    h = cy_height / z_resolution
    spheres_radius = np.max([l, h]) / 2

    if (cy_center[2] - (cy_height / 2)) < low_cylinder_limit:
        cy_center[2] = low_cylinder_limit + (cy_height / 2)

    cylinder = pv.CylinderStructured(
        center=cy_center,
        direction=cy_direction,
        radius=cy_radius,
        height=cy_height,
        theta_resolution=n_resolution,
        z_resolution=z_resolution,
    )
    cylinder_dict = {
        "mesh": cylinder,
        "center": cy_center,
        "direction": cy_direction,
        "radius": cy_radius,
        "height": cy_height,
        "theta_resolution": n_resolution,
        "z_resolution": z_resolution,
    }
    meshes["cylinder"] = cylinder_dict
    plotter.add_mesh(cylinder)

    # Create the hemispheres and add them to the faces of the cylinder
    meshes["hemispheres"] = []
    cell_count = 0
    for cell in get_geometric_objects_cell(cylinder):
        pos_cell = cell.center
        points_cell = cell.points[:3]
        norm_vec = find_normal_vector(*points_cell)
        sub_mesh = pv.Sphere(radius=spheres_radius, center=pos_cell, direction=norm_vec, phi_resolution=10, end_phi=90)
        hemisphere_dict = {
            "mesh": sub_mesh,
            "radius": spheres_radius,
            "center": pos_cell,
            "direction": norm_vec,
            "phi_resolution": 10,
            "end_phi": 90,
        }
        meshes["hemispheres"].append(hemisphere_dict)
        plotter.add_mesh(sub_mesh, show_edges=True)
    return meshes


def orient_camera_to_normal(plotter, normal):
    # Get camera parameters
    position = np.array(plotter.camera.position)
    focal_point = np.array(plotter.camera.focal_point)
    view_up = np.array(plotter.camera.up)

    # Compute new view direction
    view_dir = focal_point - position
    view_dir /= np.linalg.norm(view_dir)

    # Compute new view up
    view_up_proj = view_up - np.dot(view_up, normal) * normal
    view_up_proj /= np.linalg.norm(view_up_proj)

    # Compute new camera position
    new_position = focal_point - np.dot(view_dir, normal) * view_dir

    # Compute new focal point
    new_focal_point = new_position + view_dir

    # Update camera
    plotter.camera.position = new_position
    plotter.camera.focal_point = new_focal_point
    plotter.camera.up = view_up_proj
    return new_position


def cartesian_to_lat_lon(x, y, z):
    """
    Converts Cartesian coordinates to latitude and longitude.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arcsin(z / r)
    lon = np.arctan2(y, x)
    return np.degrees(lat), np.degrees(lon)


def compute_central_hemisphere_area(
    hemisphere_direction: ndarray,
    hemisphere_center: ndarray,
    radius_mesh: float,
    camera_radius: float,
    plotter=None,
    camera_view_angle_ccha: float = 60.0,
    near_clip_ccha: float = 1e-4,
    far_clip_ccha: float = 10.0,
) -> tuple[float, bool, list, ndarray]:
    # print("Computing central hemisphere area")

    camera_position = hemisphere_center + camera_radius * hemisphere_direction
    focal_point = camera_position - hemisphere_direction
    plotter1 = pv.Plotter()
    meshes = plotter.meshes
    for mesh in meshes:
        plotter1.add_mesh(mesh)
    # Set camera position and orientation (optional)
    plotter1.camera.clipping_range = (near_clip_ccha, far_clip_ccha)
    plotter1.camera_position = [camera_position, focal_point, (0, 0, 0)]
    plotter1.camera.view_angle = camera_view_angle_ccha
    # camera_position = orient_camera_to_normal(plotter, hemisphere_center)
    # plotter.show()

    # Get the camera's view frustum
    frustum = plotter1.camera.view_frustum()
    # if np.isinf(np.array(frustum.bounds)).any():
    #     return 0
    plotter1.add_mesh(frustum, style="wireframe")

    # Generate a plane
    direction = np.array(focal_point) - np.array(camera_position)
    direction /= np.linalg.norm(direction)

    v = hemisphere_center - np.array(camera_position)
    v /= np.linalg.norm(v)

    plane_frustum = get_plane_frustum(frustum)
    plane_eq = []

    # Get the equations of the planes of the frustum
    for plane in plane_frustum[:4]:
        a, b, c = plane[0]
        d = -(plane[0] @ plane[1])
        plane_eq.append([a, b, c, d])
        # plot_plane(plotter1, *plane)

    # Calculate the intersection between the planes of the frustum
    parametric_equation = [
        get_line_of_intersection_two_planes(plane_eq[0], plane_eq[2]),
        get_line_of_intersection_two_planes(plane_eq[0], plane_eq[3]),
        get_line_of_intersection_two_planes(plane_eq[1], plane_eq[2]),
        get_line_of_intersection_two_planes(plane_eq[1], plane_eq[3]),
    ]
    # above_plane with left_plane
    # above_plane with right_plane
    # bellow_plane with left_plane
    # bellow_plane with right_plane

    # Select the points that pass through the sphere closest to the camera
    intersection_points = []
    for p_eq in parametric_equation:
        line_eq = get_line_of_intersection_two_planes_no_sym(plane_eq[0], plane_eq[2])
        points_no_sym = np.array(line_eq(0))
        points_no_sym = np.row_stack((points_no_sym, line_eq(1)))
        intersection_test = get_intersection_points_line_sphere_no_sym(points_no_sym, (*hemisphere_center, radius_mesh))
        intersection_points_sphere = get_intersection_points_line_sphere(p_eq, (*hemisphere_center, radius_mesh))

        if len(intersection_points_sphere) == 0:
            # Test the other equations
            continue

        d1 = np.linalg.norm(intersection_points_sphere[0] - camera_position)
        d2 = np.linalg.norm(intersection_points_sphere[1] - camera_position)

        p = intersection_points_sphere[0] if d1 < d2 else intersection_points_sphere[1]
        # plotter1.add_points(p, color="green", point_size=10)
        intersection_points.append(p)
    # plotter1.show()
    # for i in range(4):
    #     distance = (np.dot(plane_eq[i][:3], hemisphere_center) + plane_eq[i][3]) / np.sqrt(plane_eq[i][0] ** 2 +
    #                                                                                        plane_eq[i][1] ** 2 +
    #                                                                                        plane_eq[i][2] ** 2)
    #     if abs(distance) < radius_mesh:
    #         print('False')
    # print('True')

    if len(intersection_points) < 3:
        # print("There is no intersection between sphere and camera frustum")
        alpha = 1 / (1 + np.linalg.norm(camera_position - hemisphere_center))
        spherical_area = 2 * alpha * np.pi * radius_mesh**2
        return spherical_area, True, plane_eq, camera_position

    for h in range(4):
        intersection_points[h] -= hemisphere_center

    # plotter1.show()
    spherical_polygon = SphericalPolygon(intersection_points)
    spherical_area = spherical_polygon.area()
    # print(f"{spherical_area=}")
    return spherical_area, False, plane_eq, camera_position


def compute_side_hemisphere_area(
    hemisphere_direction: ndarray, hemisphere_center: ndarray, radius_mesh: float, camera_radius: float, plotter=None
) -> float | bool:
    print("Computing central hemisphere area")

    camera_position = hemisphere_center + camera_radius * hemisphere_direction
    focal_point = camera_position - hemisphere_direction
    plotter1 = pv.Plotter()
    meshes = plotter.meshes
    for mesh in meshes:
        plotter1.add_mesh(mesh)
    # Set camera position and orientation (optional)
    plotter1.camera.clipping_range = (1e-4, 1)
    plotter1.camera_position = [camera_position, focal_point, (0, 0, 0)]
    # camera_position = orient_camera_to_normal(plotter, hemisphere_center)
    # plotter.show()

    # Get the camera's view frustum
    frustum = plotter1.camera.view_frustum()
    # if np.isinf(np.array(frustum.bounds)).any():
    #     return 0
    plotter1.add_mesh(frustum, style="wireframe")

    # Generate a plane
    direction = np.array(focal_point) - np.array(camera_position)
    direction /= np.linalg.norm(direction)

    v = hemisphere_center - np.array(camera_position)
    v /= np.linalg.norm(v)

    plane_frustum = get_plane_frustum(frustum)
    plane_eq = []

    # Get the equations of the planes of the frustum
    for plane in plane_frustum[:4]:
        a, b, c = plane[0]
        d = -(plane[0] @ plane[1])
        plane_eq.append([a, b, c, d])
        # plot_plane(plotter1, *plane)

    # Calculate the intersection between the planes of the frustum
    parametric_equation = [
        get_line_of_intersection_two_planes(plane_eq[0], plane_eq[2]),
        get_line_of_intersection_two_planes(plane_eq[0], plane_eq[3]),
        get_line_of_intersection_two_planes(plane_eq[1], plane_eq[2]),
        get_line_of_intersection_two_planes(plane_eq[1], plane_eq[3]),
    ]
    # above_plane with left_plane
    # above_plane with right_plane
    # bellow_plane with left_plane
    # bellow_plane with right_plane

    # Select the points that pass through the sphere closest to the camera
    intersection_points = []
    for p_eq in parametric_equation:
        intersection_points_sphere = get_intersection_points_line_sphere(p_eq, (*hemisphere_center, radius_mesh))

        if len(intersection_points_sphere) == 0:
            # Test the other equations
            continue

        d1 = np.linalg.norm(intersection_points_sphere[0] - camera_position)
        d2 = np.linalg.norm(intersection_points_sphere[1] - camera_position)

        p = intersection_points_sphere[0] if d1 < d2 else intersection_points_sphere[1]
        # plotter1.add_points(p, color="green", point_size=10)
        intersection_points.append(p)
    # plotter1.show()
    # for i in range(4):
    #     distance = (np.dot(plane_eq[i][:3], hemisphere_center) + plane_eq[i][3]) / np.sqrt(plane_eq[i][0] ** 2 +
    #                                                                                        plane_eq[i][1] ** 2 +
    #                                                                                        plane_eq[i][2] ** 2)
    #     if abs(distance) < radius_mesh:
    #         print('False')
    # print('True')

    # calculate_spherical_side_area(sphere_mesh: pv.PolyData, intersection_points: list[ndarray], plane_eq: list)
    #
    # print(f"{spherical_area=}")
    # return spherical_area, False


def intersect_plane_sphere(plane_normal, plane_point, sphere_center, sphere_radius):
    # Calculate the distance from the plane to the sphere center
    distance = np.dot(plane_normal, sphere_center - plane_point)

    # Check if the sphere is completely in front of or behind the plane
    if abs(distance) > sphere_radius:
        return []

    # Calculate the intersection points
    intersection_points = []
    if abs(distance) == sphere_radius:
        # Sphere is tangent to the plane
        intersection_points.append(sphere_center - distance * plane_normal)
    else:
        # Sphere intersects the plane
        intersection_distance = np.sqrt(sphere_radius**2 - distance**2)
        intersection_points.append(sphere_center + intersection_distance * plane_normal)
        intersection_points.append(sphere_center - intersection_distance * plane_normal)

    return intersection_points


def centroid_poly(poly: np.ndarray) -> tuple[ndarray[Any, dtype[floating[Any]]], float]:
    """
    Compute the centroid point for a Delaunay convex hull
    :param poly: Delaunay convex hull
    :return tmp_center: Geometric center position of the target object
    """
    T = Delaunay(poly).simplices
    n = T.shape[0]
    W = np.zeros(n)
    C = np.zeros(3)

    for m in range(n):
        sp = poly[T[m, :], :]
        sp += np.random.normal(0, 1e-10, sp.shape)
        W[m] = ConvexHull(sp).volume
        C += W[m] * np.mean(sp, axis=0)

    tmp_center = C / np.sum(W)
    max_distance = 0.0
    for m in range(n):
        sp = poly[T[m, :], :2]
        for spl in sp:
            distance = np.linalg.norm(spl - tmp_center[:2])
            if distance > max_distance:
                max_distance = distance

    return tmp_center, max_distance


def euler_angles_from_normal(normal_vector):
    """
    Computes Euler angles (in degrees) based on a normal vector of direction.

    Args:
    - normal_vector: A numpy array representing the normal vector of direction.

    Returns:
    - Euler angles (in degrees) as a tuple (roll, pitch, yaw).
    """
    # Normalize the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # Calculate yaw angle
    yaw = np.arctan2(normal_vector[1], normal_vector[0]) * 180 / np.pi

    # Calculate pitch angle
    pitch = np.arcsin(-normal_vector[2]) * 180 / np.pi

    # Calculate roll angle
    roll = np.arctan2(normal_vector[2], np.sqrt(normal_vector[0] ** 2 + normal_vector[1] ** 2)) * 180 / np.pi

    return yaw, pitch, roll


def point_between_planes(point, planes: ndarray):
    x, y, z = point
    count_true = 0
    for i in range(planes.shape[0]):
        for j in range(i + 1, planes.shape[0]):
            A1, B1, C1, D1 = planes[i]
            A2, B2, C2, D2 = planes[j]
            if A1 * x + B1 * y + C1 * z + D1 < 0 and A2 * x + B2 * y + C2 * z + D2 > 0:
                count_true += 1
            if A1 * x + B1 * y + C1 * z + D1 > 0 and A2 * x + B2 * y + C2 * z + D2 < 0:
                count_true += 1
    if count_true >= 2:
        return True
    else:
        return False


def get_side_hemisphere_area(
    count_plane_gsha: int, meshes_gsha: dict, frustum_planes: list, central_hemisphere_gsha: int, n_resolution: int
) -> float:
    tmpidxs = 49 * [[]]
    number_of_elements = 0
    tmpidxs[number_of_elements] = central_hemisphere_gsha
    number_of_elements += 1
    for count_idx in range(1, 3):
        tmpidxs[number_of_elements] = (central_hemisphere_gsha + count_idx) % n_resolution + (
            central_hemisphere_gsha // n_resolution
        ) * n_resolution
        number_of_elements += 1
        tmpidxs[number_of_elements] = (central_hemisphere_gsha - count_idx) % n_resolution + (
            central_hemisphere_gsha // n_resolution
        ) * n_resolution
        number_of_elements += 1
    list_idx = tmpidxs.copy()
    total_elements = number_of_elements
    if central_hemisphere_gsha > n_resolution:
        for l in range(total_elements):
            list_idx[number_of_elements] = list_idx[l] - n_resolution
            number_of_elements += 1
    tmpidxs = list_idx.copy()
    total_elements = number_of_elements
    if central_hemisphere_gsha < count_plane_gsha - n_resolution:
        for l in range(total_elements):
            list_idx[number_of_elements] = list_idx[l] + n_resolution
            number_of_elements += 1

    list_idx = list_idx[:number_of_elements]
    area = 0
    for hemisphere_idx in list_idx[1:]:
        ct_pt = np.array(meshes_gsha["hemispheres"][hemisphere_idx]["center"])
        is_in = False
        intersection_points = []
        for plane_gsha in frustum_planes:
            distance = abs(
                np.dot(plane_gsha[:3], meshes_gsha["hemispheres"][hemisphere_idx]["center"]) + plane_gsha[3]
            ) / np.sqrt(plane_gsha[0] ** 2 + plane_gsha[1] ** 2 + plane_gsha[2] ** 2)
            if distance < meshes_gsha["hemispheres"][hemisphere_idx]["radius"]:
                x = (
                    -plane_gsha[3]
                    - meshes_gsha["hemispheres"][hemisphere_idx]["center"][1] * plane_gsha[1]
                    - meshes_gsha["hemispheres"][hemisphere_idx]["center"][2] * plane_gsha[2]
                ) / plane_gsha[0]
                point_pi = np.array(
                    [
                        x,
                        meshes_gsha["hemispheres"][hemisphere_idx]["center"][1],
                        meshes_gsha["hemispheres"][hemisphere_idx]["center"][2],
                    ]
                )
                intersection_points = intersect_plane_sphere(
                    np.array(plane_gsha[:3]),
                    point_pi,
                    np.array(meshes_gsha["hemispheres"][hemisphere_idx]["center"]),
                    meshes_gsha["hemispheres"][hemisphere_idx]["radius"],
                )
                is_in = True
                break
        alpha = 1
        if not is_in:
            if not point_between_planes(ct_pt, np.array(frustum_planes)):
                area += 2 * alpha * np.pi * meshes_gsha["hemispheres"][hemisphere_idx]["radius"] ** 2
            else:
                area += 0
        else:
            if point_between_planes(ct_pt, np.array(frustum_planes)):
                area += (
                    alpha
                    * 2
                    * np.pi
                    * meshes_gsha["hemispheres"][hemisphere_idx]["radius"]
                    * np.linalg.norm(intersection_points[0] - intersection_points[1])
                )
            else:
                area += alpha * (
                    2
                    * np.pi
                    * meshes_gsha["hemispheres"][hemisphere_idx]["radius"]
                    * np.linalg.norm(intersection_points[0] - intersection_points[1])
                    + 2 * np.pi * meshes_gsha["hemispheres"][hemisphere_idx]["radius"]
                )
    return area


def points_along_line(start_point, end_point, num_points):
    """
    Returns points in a line on 3D space
    :param start_point: Start point of the line
    :param end_point:  End point of the line
    :param num_points: Number of points between start and end point
    :return points: The points in the line
    """
    # Generate num_points equally spaced between start_point and end_point
    x = np.linspace(start_point[0], end_point[0], num_points)
    y = np.linspace(start_point[1], end_point[1], num_points)
    z = np.linspace(start_point[2], end_point[2], num_points)
    points = np.column_stack((x, y, z))
    return points


def is_point_inside(point, hull):
    """
    Verify is a point is inside a Delaunay convex hull
    :param point: Point to be evaluated
    :param hull: The convex hull computed by Delaunay function of Scipy
    :return point_in_hull: Boolean denoting if point is inside the hull True=Yes, False=No
    """
    # Check if the given point is within the convex hull
    point_in_hull = hull.find_simplex(point) >= 0
    return point_in_hull


def is_line_through_convex_hull(hull, line):
    """
    Verify if a line pass by a Delaunay convex hull
    :param hull: he convex hull computed by Delaunay function of Scipy
    :param line: Points on a line
    :return: Boolean denoting if line goes through the hull True=Yes, False=No
    """
    for point in line:
        if is_point_inside(point, hull):
            return True
    return False


def get_rotation_quat(curr_pos, target_pos):
    """
    Calculates the quaternion representing the rotation needed to align the current position
    to face the target position.

    The function computes a "look-at" vector from the current position to the target position,
    then calculates the corresponding rotation matrix. The matrix is converted into a quaternion
    to represent the 3D rotation.

    Parameters:
    ----------
    curr_pos : array-like
        The current position as a 3D vector (x, y, z).

    target_pos : array-like
        The target position as a 3D vector (x, y, z).

    Returns:
    --------
    quaternion : np.ndarray
        A 4-element array representing the rotation as a quaternion [x, y, z, w].

    Notes:
    ------
    - The "look-at" vector is normalized to get the direction from the current position
      to the target.
    - The "up" vector is assumed to be [0, 0, 1], which is aligned with the Z-axis.
    - The right and new up vectors are calculated via cross products to form an orthogonal
      coordinate system, which is then used to create the rotation matrix.
    - The `Rot.from_matrix()` function from `scipy.spatial.transform` is used to convert the
      rotation matrix into a quaternion.
    """
    look_at = np.array(target_pos) - np.array(curr_pos)
    look_at = look_at / np.linalg.norm(look_at)

    up = np.array([0.0, 0.0, 1.0])

    right = np.cross(up, look_at)
    right = right / np.linalg.norm(right)

    new_up = np.cross(look_at, right)

    rotation_matrix = np.array([right, new_up, look_at]).T

    rotation = Rot.from_matrix(rotation_matrix)

    return rotation.as_quat()


def horn(P: ndarray, Q: ndarray) -> ndarray:
    """Calculate the transformation matrix using Horn's method"""
    if P.shape != Q.shape:
        raise RuntimeError("Matrices P and Q must be of the same dimensionality")

    # Calculate the centroids of the points
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Subtract centroids from the points
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Compute the covariance matrix
    H = np.dot(P_centered.T, Q_centered)

    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute the rotation matrix
    R = np.dot(Vt.T, U.T)

    # Ensure a proper rotation (det(R) == 1)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute the scale factor
    scale = np.sum(S) / np.sum(P_centered**2)

    # Compute the translation
    t = centroid_Q - scale * np.dot(R, centroid_P)

    # Create the homogeneous transformation matrix
    T = np.identity(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t

    return T


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    pos_mesh = np.array([-2, 0, 0])
    r_mesh = 2

    # cam_pos = np.array((2.9, -0.1, 0.0))

    cy_direction = np.array([0, 0, 1])
    cy_height = 4
    n_resolution = 24

    # Create a plotter
    plotter = pv.Plotter()
    meshes = draw_cylinder_with_hemisphere(
        plotter, cy_direction, cy_height, n_resolution, 2.0, np.array([0.0, 0.0, 0.0]), 0.0
    )
    first_hemisphere = meshes["hemispheres"][20]
    cam_pos = np.array(first_hemisphere["direction"])
    pos_mesh = np.array(first_hemisphere["center"])
    r_mesh = first_hemisphere["radius"]
    compute_central_hemisphere_area(cam_pos, pos_mesh, r_mesh, plotter, 60.0, 1e-4, 10)
    plotter.show()
    # ax = plot_circle(1.0, 500)
    # vector = np.array((0, 1, 1))
    # point = np.array((0, 0, 0))
    # radius = 1.0
    # ax = plot_circle_in_plane(vector, point, radius)
    # Example usage
    # point1 = np.array((2, 2, 3))
    # point2 = np.array((4, 5, 6))
    # point3 = np.array((7, 7, 9))
    # get_viewed_area()
    # normal, ax = plot_plane_through_points(point1, point2, point3)
    # ax = plot_circle_in_plane(normal, point1, radius, ax)
    # ax = plot_line(normal, point1, ax)
    # ax, x, y, z = plot_hemisphere(point1, radius, 0.0, ax)
    # rot = compute_orientation(normal)
    # for i in range(x.shape[0]):
    #     Points = np.column_stack((x[:, i], y[:, i], z[:, i]))
    #     rot_points = rot.apply(Points)
    #     ax.scatter(rot_points[:, 0], rot_points[:, 1], rot_points[:, 2])
    # plt.show()
    # get_viewed_area()  # Only function used with pyvista

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
    # pi1gl = np.array([2.0, 4.0, -1.0, 1.0])
    # pi2gl = np.array([-1.0, 2.0, 1.0, 2.0])
    # parametric_equation = get_line_of_intersection_two_planes(pi1gl, pi2gl)
    # print(parametric_equation)
    # xl = parametric_equation[0].subs(t, 0.0)
    # yl = parametric_equation[1].subs(t, 0.0)
    # zl = 0.0
    # sphere_eq = (xl, yl, zl, 2)
    # Find intersection point(s)
    # intersection_points = get_intersection_points_line_sphere(parametric_equation, sphere_eq)

    # Display intersection point(s)
    # print("Intersection point(s) with the sphere:")
    # for point in intersection_points:
    #     print(point)

    # distance = np.linalg.norm(intersection_points[0] - intersection_points[1])

    # print(distance)
