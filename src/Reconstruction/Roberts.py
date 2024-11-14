import numpy as np
import pyvista as pv

import Config
from numpy import ndarray
from GeometryOperations import find_normal_vector, get_geometric_objects_cell

settings = Config.Settings.get()

def rdraw_cylinder_with_hemisphere(
    plotter,
    cy_direction: ndarray,
    cy_height: float,
    theta_resolution: int,
    cy_radius: float,
    cy_center: ndarray,
    low_cylinder_limit=0.0,
):
    print("Drawing cylinder with hemispheres")
    meshes = {}

    # Calculate the length of the lateral surface of an inscribed cylinder
    spheres_radius = np.sin(np.pi / theta_resolution) * cy_radius
    z_resolution = int(np.round(cy_height / spheres_radius)) + 1
    cy_height = 1.15 * (z_resolution - 1) * spheres_radius
    cy_center[2] = low_cylinder_limit + (cy_height / 2)

    cylinder = pv.CylinderStructured(
        center=cy_center,
        direction=cy_direction,
        radius=cy_radius,
        height=cy_height,
        theta_resolution=theta_resolution,
        z_resolution=z_resolution,
    )
    cylinder_dict = {
        "mesh": cylinder,
        "center": cy_center,
        "direction": cy_direction,
        "radius": cy_radius,
        "height": cy_height,
        "theta_resolution": theta_resolution,
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
        sub_mesh = pv.Sphere(radius=spheres_radius, center=pos_cell, direction=norm_vec, phi_resolution=5, theta_resolution=10, end_phi=90)
        hemisphere_dict = {
            "mesh": sub_mesh,
            "radius": spheres_radius,
            "center": pos_cell,
            "direction": norm_vec,
            "phi_resolution": 5,
            "theta_resolution": 10,
            "end_phi": 90,
        }
        meshes["hemispheres"].append(hemisphere_dict)
        plotter.add_mesh(sub_mesh, show_edges=True)
    return meshes


def draw_cylinders_with_hemispheres(centroid_points, radius, target_points):
    """
    Draw the hemispheres and the cylinders around the object
    :param centroid_points_pf: Dictionary of arrays of points with the central points of the objects
    :param radius_pf: Computed radius of the cylinders around each object
    :param target_points_pf: Computed points for the convex hull of the objects
    :return: vector_of_points: Dictionary with points around each object
    :return: Dictionary for weight of each point
    """
    print("Starting showing data")

    height_proportion = float(settings["height proportion"])
    n_resolution = int(settings["n resolution"])

    # Create a plotter
    plotter = pv.Plotter()
    meshes = {}

    for target in centroid_points.keys():
        cy_direction = np.array([0, 0, 1])
        cy_hight = height_proportion * (np.max(target_points[target][:, 2]) - np.min(target_points[target][:, 2]))
        r_mesh = radius[target]

        # Find the radius of the spheres
        target_meshes = rdraw_cylinder_with_hemisphere(
            plotter, cy_direction, cy_hight, n_resolution, r_mesh, centroid_points[target], 0.0
        )
        meshes[target] = target_meshes

    # plotter.show()

    return plotter, meshes


def generate_target_view_points(target_meshes, radius):
    hight_step = 4 + 1
    theta_step = np.deg2rad(15)
    tilt = np.deg2rad(-5)

    target_points = {}
    list_points = []

    for target in target_meshes:
        start_radius = 0.4 * radius[target]
        step_radius_point = 0.4 * radius[target]
        max_radius = start_radius + (1 + 0.1) * step_radius_point

        if len(target_points) == 0:
            target_points[target] = [np.array([-2.0, 0.0, 1.0, 0.0, 0.0, 0.0])]
            list_points.append(np.array([-2.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
        else:
            target_points[target] = []

        cylinder_radius = target_meshes[target]["cylinder"]["radius"]
        cylinder_center = target_meshes[target]["cylinder"]["center"]

        hight_step_points = 2 * cylinder_center[2] / hight_step

        for z in np.arange(hight_step_points, 2 * cylinder_center[2], hight_step_points):
            if np.isclose(z, 2 * cylinder_center[2], atol=1e-6):
                continue

            for r in np.arange(cylinder_radius + start_radius, cylinder_radius + max_radius, step_radius_point):
                if np.isclose(r, max_radius, atol=1e-6):
                    continue

                for theta in np.arange(0, 2 * np.pi, theta_step):
                    if np.isclose(theta, 2 * np.pi, atol=1e-6):
                        continue

                    x = np.cos(theta) * r + cylinder_center[0]
                    y = np.sin(theta) * r + cylinder_center[1]

                    pan = theta + np.pi if theta + np.pi >= 2 * np.pi else theta - np.pi
                    target_points[target].append(np.array([x, y, z, pan, tilt, 0]))
                    list_points.append(np.array([x, y, z, pan, tilt, 0]))

        target_points[target] = np.array(target_points[target])

    return target_points, np.array(list_points)


def is_cell_visible(cell, camera_position):
    normal_vector = find_normal_vector(*cell.points)
    vector2face = np.array(camera_position) - np.array(cell.center)

    return normal_vector @ vector2face > 0


def is_point_close_frustum(point, camera_position, frustum, threshold=1e-8):
    normal_faces = frustum.face_normals[:4]

    # Get the equations of the planes of the frustum
    plane_eq = []
    for normal in normal_faces:
        d = -(np.array(normal) @ np.array(camera_position))
        plane_eq.append((lambda p, n, d: np.array(n) @ np.array(p) + d + threshold, normal, d))

    return all([fn(point, n, d) > 0 for fn, n, d in plane_eq])


def is_line_through_cylinder(cylinder, point_a, point_b):
    line_vector = np.array(point_a) - np.array(point_b)
    line_distance = np.linalg.norm(line_vector[:2])

    cylinder_position = cylinder["center"][:2]
    cylinder_radius = cylinder["radius"]
    camera_position = point_a[:2]

    camera_cylinder_distance = np.linalg.norm(np.array(cylinder_position) - np.array(camera_position))
    minimal_value = np.sqrt(camera_cylinder_distance**2 - cylinder_radius**2)

    if minimal_value > line_distance:
        return False

    dx = point_a[0] - cylinder_position[0]
    dy = point_a[1] - cylinder_position[1]

    A = line_vector[0] ** 2 + line_vector[1] ** 2
    B = 2 * (line_vector[0] * dx + line_vector[1] * dy)
    C = dx**2 + dy**2 - cylinder_radius**2

    return B**2 - 4 * A * C >= 0


def is_hemisphere_inside_frustum(cylinder, camera_position: tuple, hemisphere: dict, frustum):
    hemisphere_center = hemisphere["center"]
    hemisphere_radius = hemisphere["radius"]
    hemisphere_direction = hemisphere["direction"]
    threshold = hemisphere_radius / 2
    target_point = hemisphere_center + hemisphere_direction * hemisphere_radius / 2

    v = np.array(camera_position[0]) - np.array(camera_position[1])
    w = np.array(hemisphere_direction)
    if v @ w <= 0:
        return False

    if is_line_through_cylinder(cylinder, camera_position[0], target_point):
        return False

    return is_point_close_frustum(target_point, camera_position[0], frustum, threshold)


def get_hemisphere_in_frustum(cylinder, camera_position: tuple, hemispheres: list, frustum):
    hemisphere_in_frustum = []

    for hemisphere in hemispheres:
        if is_hemisphere_inside_frustum(cylinder, camera_position, hemisphere, frustum):
            hemisphere_in_frustum.append(hemisphere)

    return hemisphere_in_frustum


def reward_point(capture_area, cylinder_distance, cylinder_radius):
    k = np.sqrt(cylinder_radius)
    d = 1 / (np.sqrt(2) * cylinder_radius)

    reward = lambda x: (d * x) ** k / (1 + (d * x) ** (k + 1))

    return capture_area * reward(cylinder_distance)


def compute_points_weights(target_view_points: dict, target_meshes: dict, plotter):
    print("Starting compute weights ...")
    target_view_weights = {}

    for target, points in target_view_points.items():
        print("Computing weights for hemisphere")

        if len(target_view_weights) == 0:
            points = points[1:]
            target_view_weights[target] = [0]
        else:
            target_view_weights[target] = []

        cylinder_hemispheres = target_meshes[target]["hemispheres"]
        target_cylinder = target_meshes[target]["cylinder"]
        cylinder_center = target_cylinder["center"]
        cylinder_radius = target_cylinder["radius"]

        for i, point in enumerate(points):

            camera_position = [point[0], point[1], point[2]]
            focal_point = [cylinder_center[0], cylinder_center[1], point[2]]
            distance_camera = np.linalg.norm(np.array(camera_position[:2]) - np.array(focal_point[:2]))
            up_vector = (0.0, 0.0, 1.0)

            # tilt camera
            focal_point[2] += np.tan(point[4]) * np.linalg.norm(np.array(camera_position) - np.array(focal_point))

            plotter.camera_position = (camera_position, focal_point, up_vector)
            plotter.camera.view_angle = 82.6
            plotter.camera.clipping_range = (1e-5, 10)

            frustum = plotter.camera.view_frustum()

            target_hemispheres = get_hemisphere_in_frustum(
                target_cylinder, plotter.camera_position, cylinder_hemispheres, frustum
            )

            camera_area = 0
            for hemisphere in target_hemispheres:
                print(f"\r\033[Kpoint [{i + 1}:{len(points)}], hemisphere: len({len(target_hemispheres)})", end="")

                for cell in get_geometric_objects_cell(hemisphere["mesh"]):
                    if not is_point_close_frustum(cell.center, camera_position, frustum):
                        continue

                    if not is_cell_visible(cell, camera_position):
                        continue

                    vector_v = np.array(cell.points[1]) - np.array(cell.points[0])
                    vector_w = np.array(cell.points[2]) - np.array(cell.points[0])

                    # cell area
                    camera_area += np.linalg.norm(np.cross(vector_v, vector_w)) / 2

                    # mesh = pv.Line(camera_position, cell.center)
                    # plotter.add_mesh(mesh, color="k", line_width=2)

            target_view_weights[target].append(reward_point(camera_area, distance_camera, cylinder_radius))

            # frustum_actor = plotter.add_mesh(frustum, style="wireframe")
            # plotter.show()
            # plotter.remove_actor(frustum_actor)

        target_view_weights[target] = np.array(target_view_weights[target])
        print("\r\033[K", end="")

    return target_view_weights
