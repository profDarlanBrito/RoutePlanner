import ast
import csv
import datetime
import os
import pickle
from typing import Any

import cv2 as cv
import numpy as np
from numpy import bool_, dtype, ndarray

import Config
from CoppeliaInterface import CoppeliaInterface
from GeometryOperations import euler_angles_from_normal, get_rotation_quat

settings = Config.Settings.get()


def get_result_cops_route(result_cops_path: str, points_by_id: dict):
    with open(result_cops_path, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=";")
        _ = next(reader)  # header
        line = next(reader)  # first line content
        line = line[7].replace("  ", ", ")
        cops_route = ast.literal_eval(line)

    route = [points_by_id[cops_route[0][0]]]
    route_id = [cops_route[0][0]]

    for _, key_id in cops_route:
        route.append(points_by_id[key_id])
        route_id.append(key_id)

    return route, route_id


def calculate_position_orientation(find_point: list, tilt_angle_deg: float = -5) -> np.ndarray:
    """
    Auxiliary function to calculate the position and orientation based on the current position and camera angle.

    Parameters:
    - find_point: List containing the coordinates (x, y, z) and the angle (theta) of the point.
    - tilt_angle_deg: Camera tilt angle in degrees (default: -5).

    Returns:
    - An array with the position and orientation of the point.
    """
    theta = np.deg2rad(find_point[3])
    curr_pos = find_point[:3]

    tilt_camera = np.deg2rad(tilt_angle_deg)

    target_pos = np.copy(curr_pos)
    target_pos[0] += np.cos(theta)
    target_pos[1] += np.sin(theta)
    target_pos[2] += np.tan(tilt_camera) * np.linalg.norm(curr_pos - target_pos)

    ori = get_rotation_quat(curr_pos, target_pos)
    return np.hstack((curr_pos, ori))


def get_result_by_cluster(points_by_id: dict, cops_by_id: list, interval: dict):
    cluster = {}

    for point_id in cops_by_id:
        # ignore initial drone position
        if point_id == 0:
            continue

        for group_name, (min, max) in interval.items():
            if min <= point_id <= max:
                if group_name not in cluster:
                    cluster[group_name] = []

                find_point = points_by_id[point_id]
                pos_ori = calculate_position_orientation(find_point)

                cluster[group_name].append(pos_ori)
                break

    return cluster


def compute_route_distance(route: list):
    distance = 0.0

    for i, p in enumerate(route[1:]):
        distance += np.linalg.norm(route[i][:3] - p[:3])  # p == route[i + 1]

    return distance


def read_route_cops(result_cops_path: str, interval: dict, conversion_table: list):

    cops_route, cops_by_id = get_result_cops_route(result_cops_path, conversion_table)

    cops_route_by_group = get_result_by_cluster(conversion_table, cops_by_id, interval)

    route_distace = compute_route_distance(cops_route)

    return route_distace, cops_route, cops_route_by_group


def generate_true_spiral_points(radius: float, num_points: int) -> list[ndarray]:
    """
    Generates a 3D spiral with a specified number of points.

    The spiral is generated in the XY plane, with each point adjusted to form a smooth curve
    around an imaginary cylinder with the given radius. The height of each point is calculated
    to form a helical curve that fits within a sphere of the specified radius.

    Parameters:
    ----------
    radius : float
        The radius of the spiral around the Z-axis.

    num_points : int
        The number of points to generate along the spiral.

    Returns:
    --------
    points : list of np.ndarray
        A list of arrays with coordinates (x, y, z) representing the points along
        the spiral.

    Notes:
    ------
    - The height of each point 'z' is calculated according to the sphere equation
      (x^2 + y^2 + z^2 = radius^2) to ensure the points are distributed in a spiral
      within the sphere.
    - The constant 'c' is used to adjust the shape of the spiral, while 'k' defines
      the scaling factor based on 'c'.
    """
    c = 12
    k = c * np.pi

    radius *= 0.85

    step = k**2 / (num_points + 1)

    r = lambda theta: radius * theta / k
    height = lambda x, y: np.sqrt(radius**2 - x**2 - y**2)

    points = []
    for curr_step in np.arange(step, k**2, step):
        theta = np.sqrt(curr_step)

        x = r(theta) * np.cos(theta)
        y = r(theta) * np.sin(theta)
        z = height(x, y)

        points.append(np.array([x, y, z]))

    return points


def get_single_target_true_spiral_trajectory(
    centroid_points_gstst: ndarray, radius_gstst: float, num_points_gstst: float
):
    print("Generating spiral trajectory over one target")

    points_gstst = generate_true_spiral_points(radius_gstst, num_points_gstst)

    points_gstst = [point + centroid_points_gstst for point in points_gstst]

    directions_gstst = [get_rotation_quat(curr_pos, centroid_points_gstst) for curr_pos in points_gstst]

    return np.array(points_gstst), np.array(directions_gstst)


def generate_spiral_points(box_side_gsp, step):
    x, y = 0, 0
    points = []

    directions = [(step, 0), (0, step), (-step, 0), (0, -step)]  # Right, Up, Left, Down
    direction_index = 0
    steps = 1
    step_count = 0

    while True:
        dx, dy = directions[direction_index % 4]
        x += dx
        y += dy
        points.append([x, y])
        step_count += 1

        if step_count == steps:
            direction_index += 1
            step_count = 0
            if direction_index % 2 == 0:
                steps += 1

        if not abs(x) < box_side_gsp / 2 or not abs(y) < box_side_gsp / 2:
            points.pop()
            break

    return points


def get_single_target_spiral_trajectory(centroid_points_gstst: ndarray, radius_gstst: float, parts_gstst: float):
    print("Generating spiral trajectory over one target")
    step = radius_gstst / parts_gstst
    points_gstst = generate_spiral_points(radius_gstst, step)

    points_gstst = np.array([np.hstack((np.array(p), 0)) for p in points_gstst])
    directions_gstst = np.zeros([1, 3])
    for p in points_gstst[1:]:
        directions_gstst = np.row_stack((directions_gstst, euler_angles_from_normal(-p)))
    points_gstst = points_gstst + centroid_points_gstst

    return points_gstst, directions_gstst


def get_spiral_trajectories(
    centroids_gst: dict, radius_gst: dict, parts_gst: int
) -> tuple[ndarray[Any, dtype[Any]], dict[Any, ndarray[Any, dtype[bool_]]], dict[Any, Any], int]:
    print("Generating spiral trajectories")
    # plotter_gst = pv.Plotter()
    route_gst = np.zeros([1, 7])
    route_by_target_gst = {}
    spiral_target_distance_gst = {}
    total_distance_gst = 0
    for target_gst, centroid_gst in centroids_gst.items():
        radius_box_gst = 3 * radius_gst[target_gst]
        # centroid_gst[2] = centroid_gst[2] + scale_to_height_spiral * centroid_gst[2]
        centroid_gst[2] /= 2
        spiral_point, spiral_direction = get_single_target_true_spiral_trajectory(
            centroid_gst, radius_box_gst, parts_gst
        )
        spiral_target_distance_gst[target_gst] = 0
        for count_point in range(spiral_point.shape[0] - 1):
            spiral_target_distance_gst[target_gst] += np.linalg.norm(
                spiral_point[count_point, :3] - spiral_point[count_point + 1, :3]
            )
        route_by_target_gst[target_gst] = np.column_stack((spiral_point, spiral_direction))
        route_gst = np.row_stack((route_gst, route_by_target_gst[target_gst]))
        total_distance_gst += spiral_target_distance_gst[target_gst]
    route_gst = np.row_stack((route_gst, np.zeros([1, 7])))
    # Create lines connecting each pair of adjacent points
    # lines = []
    # for i in range(route_gst.shape[0] - 1):
    #     lines.append([2, i, i + 1])  # Each line is represented as [num_points_in_line, point1_index, point2_index]
    # lines = np.array(lines, dtype=np.int_)

    # plotter_gst.add_mesh(pv.PolyData(route_gst[:, :3]))
    # plotter_gst.add_mesh(pv.PolyData(route_gst[:, :3], lines=lines))
    #
    # plotter_gst.show_grid()
    # plotter_gst.show()
    return route_gst, route_by_target_gst, spiral_target_distance_gst, total_distance_gst


def get_random_points(interval: dict, conversion_table: list, targets_points_of_view: dict) -> dict:
    print("Generating random trajectories")

    target_select_point_view = {}
    list_canditates = []
    for min, max in interval.values():
        if min == 0:
            min = 1
        list_canditates.append(np.arange(min, max + 1))

    for (target, _), canditates in zip(targets_points_of_view.items(), list_canditates):
        print("Generating random trajectory over one target")

        length = len(canditates)
        target_select_point_view[target] = []

        p = np.abs(np.random.normal(0.6, 0.1))
        p = 0.1 if p < 0.1 else p
        p = 1.0 if p > 1 else p

        pick = int(length * p)
        points_views_index = np.sort(np.random.choice(canditates, pick, replace=False))

        for index in points_views_index:
            find_point = conversion_table[index]
            pos_ori = calculate_position_orientation(find_point)

            target_select_point_view[target].append(pos_ori)

    target_distance = {}
    for target, route in target_select_point_view.items():
        if target not in target_distance:
            target_distance[target] = []

        target_distance[target] = compute_route_distance(route)

    return target_select_point_view, target_distance


def get_image(sim, sequence: int, file_name: str, vision_handle: int, directory_name_gi: str):
    """
    Method used to get the image from vision sensor on coppeliaSim and save the image in a file.
    The vision handle must be previously loaded.
    :param vision_handle: Vison sensor handle to CoppeliaSim vision sensor.
    :param file_name: File name to saved image
    :param sequence: Parameter not used yet
    :return: Nothing
    """
    img, resolution = sim.getVisionSensorImg(vision_handle)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)

    # Camera position [x y z]
    position = sim.getObjectPosition(vision_handle, sim.handle_world)

    # Camera orientation Euler angles [qx qy qz qw]
    orientarion = sim.getObjectQuaternion(vision_handle, sim.handle_world)

    # Define the directory name
    directory_name = directory_name_gi

    # Specify the path where you want to create the directory
    path = settings["path"]  # You can specify any desired path here

    # Construct the full path
    full_path = os.path.join(path, directory_name)

    # Check if the directory already exists
    if not os.path.exists(full_path):
        # Create the directory
        os.makedirs(full_path)

    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Extract individual components
    year = str(current_datetime.year)
    month = str(current_datetime.month)
    day = str(current_datetime.day)
    hour = str(current_datetime.hour)
    minute = str(current_datetime.minute)

    image_name = (
        file_name
        + "_"
        + day
        + "_"
        + month
        + "_"
        + hour
        + "_"
        + minute
        + "_"
        + str(sequence)
        + "."
        + settings["extension"]
    )
    filename = os.path.join(full_path, image_name)

    ref_image_path = os.path.join(full_path, "ref_images.txt")

    # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
    # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
    # and color format is RGB triplets, whereas OpenCV uses BGR:
    img = cv.flip(cv.cvtColor(img, cv.COLOR_BGR2RGB), 0)

    cv.imwrite(filename, img)

    with open(ref_image_path, "a") as file:
        file.write(
            f"{image_name} {position[0]} {position[1]} {position[2]} {orientarion[3]} {orientarion[0]} {orientarion[1]} {orientarion[2]}\n"
        )


def quadcopter_control(sim, client, quad_target_handle, quad_base_handle, route_qc: dict):
    """
    This method is used to move the quadcopter in the CoppeliaSim scene to the position pos.
    :param route_qc:
    :param client:
    :param sim:
    :param quad_base_handle: The handle to get the quadcopter current position
    :param quad_target_handle:  The handle to the target of the quadcopter. This handle is used to position give the
    position that the quadcopter must be after control.
    :return: A boolean indicating if the quadcopter reach the target position.
    """
    for target, position_orientation in route_qc.items():
        for i in range(position_orientation.shape[0]):
            cone_name = "./" + target + f"/Cone"
            handle = sim.getObject(cone_name, {"index": i, "noError": True})
            if handle < 0:
                break
            cone_pos = list(position_orientation[i, :3])
            sim.setObjectPosition(handle, cone_pos)
        for each_position in position_orientation:
            pos = list(each_position[:3])
            next_point_handle = sim.getObject("./new_target")
            sim.setObjectPosition(next_point_handle, pos)
            orientation = list(np.deg2rad(each_position[3:]))
            orientation_angles = [0.0, 0.0, orientation[0]]
            # sim.setObjectOrientation(quad_target_handle, [0.0, 0.0, orientation[0]], sim.handle_world)
            # while sim.getSimulationTime() < t_stab:
            #     print(sim.getObjectOrientation(quad_base_handle, sim.handle_world))
            #     client.step()
            #     continue
            # orientation_angles = sim.yawPitchRollToAlphaBetaGamma(orientation[0], orientation[2], orientation[1])
            # pos = sim.getObjectPosition(quad_base_handle, sim.handle_world)
            # camera_handle = sim.getObject('./O[0]/Cone[19]')
            # sim.setObjectOrientation(camera_handle, orientation_angles)
            # sim.setObjectPosition(camera_handle, pos)
            # client.step()
            # sim.setObjectOrientation(quad_target_handle, sim.handle_world, orientation_angles)
            total_time = sim.getSimulationTime() + settings["total simulation time"]
            stabilized = False
            while sim.getSimulationTime() < total_time:
                diff_pos = np.subtract(pos, sim.getObjectPosition(quad_base_handle, sim.handle_world))
                norm_diff_pos = np.linalg.norm(diff_pos)
                if norm_diff_pos > 0.5:
                    delta_pos = 0.1 * diff_pos
                    new_pos = list(sim.getObjectPosition(quad_base_handle, sim.handle_world) + delta_pos)
                else:
                    new_pos = pos

                sim.setObjectPosition(quad_target_handle, new_pos)
                diff_ori = np.subtract(orientation_angles, sim.getObjectOrientation(quad_base_handle, sim.handle_world))
                norm_diff_ori = np.linalg.norm(diff_ori)

                if norm_diff_ori > 0.08:
                    delta_ori = 0.3 * diff_ori
                    new_ori = list(sim.getObjectOrientation(quad_base_handle, sim.handle_world) + delta_ori)
                else:
                    new_ori = orientation_angles
                sim.setObjectOrientation(quad_target_handle, new_ori)
                t_stab = sim.getSimulationTime() + settings["time to stabilize"]
                while sim.getSimulationTime() < t_stab:
                    diff_pos = np.subtract(new_pos, sim.getObjectPosition(quad_base_handle, sim.handle_world))
                    diff_ori = np.subtract(new_ori, sim.getObjectOrientation(quad_base_handle, sim.handle_world))
                    norm_diff_pos = np.linalg.norm(diff_pos)
                    norm_diff_ori = np.linalg.norm(diff_ori)
                    if norm_diff_pos < 0.1 and norm_diff_ori < 0.05:
                        stabilized = True
                        break
                    client.step()
                diff_pos = np.subtract(pos, sim.getObjectPosition(quad_base_handle, sim.handle_world))
                diff_ori = np.subtract(orientation_angles, sim.getObjectOrientation(quad_base_handle, sim.handle_world))
                norm_diff_pos = np.linalg.norm(diff_pos)
                norm_diff_ori = np.linalg.norm(diff_ori)
                if norm_diff_pos < 0.1 and norm_diff_ori < 0.05:
                    stabilized = True
                    break
                client.step()
            if not stabilized:
                print("Time short")


def quadcopter_control_direct_points(
    sim, client, vision_handle: int, route_qc: ndarray, filename_qcdp: str, directory_name_qcdp: str
):
    """
    This method is used to move the quadcopter in the CoppeliaSim scene to the position pos.
    :param route_qc:
    :param client:
    :param sim:
    :param quad_base_handle: The handle to get the quadcopter current position
    :param quad_target_handle:  The handle to the target of the quadcopter. This handle is used to position give the
    position that the quadcopter must be after control.
    :return: A boolean indicating if the quadcopter reach the target position.
    """
    count_image = 0

    for point_qcdp in route_qc:
        pos = list(point_qcdp[:3])
        ori = list(point_qcdp[3:])

        sim.setObjectPosition(vision_handle, pos, sim.handle_world)
        sim.setObjectQuaternion(vision_handle, ori, sim.handle_world)

        total_time = sim.getSimulationTime() + settings["total simulation time"]
        while sim.getSimulationTime() < total_time:
            client.step()

        get_image(sim, count_image, filename_qcdp, vision_handle, directory_name_qcdp)
        count_image += 1


def view_point(copp: CoppeliaInterface, experiment: int):
    with open(os.path.join(settings["save path"], f"variables/convex_hull_{experiment}.var"), "rb") as file:
        S = pickle.load(file)
        targets_points_of_view = pickle.load(file)
        centroid_points = pickle.load(file)
        radius = pickle.load(file)
        conversion_table = pickle.load(file)
        interval = pickle.load(file)

    result_cops_path = os.path.join(settings["COPS result"], f"{settings['COPS problem']}{str(experiment)}.csv")
    # result_op_path = os.path.join(settings["COPS result"], f"{settings['OP problem']}{str(experiment)}.csv")

    cops_route_distace, cops_route, cops_route_by_group = read_route_cops(result_cops_path, interval, conversion_table)
    # op_route_distace, op_route, op_route_by_group = read_route_cops(result_op_path, interval, conversion_table)

    parts_to_spiral = 100
    spiral_routes, spiral_route_by_target, spiral_target_distance, travelled_spiral_distance = get_spiral_trajectories(
        centroid_points, radius, parts_to_spiral
    )

    random_route_by_target, random_target_distance = get_random_points(
        interval, conversion_table, targets_points_of_view
    )

    copp.handles[settings["vision sensor names"]] = copp.sim.getObject(settings["vision sensor names"])
    # vision_handle = copp.handles[settings['vision sensor names']]
    # filename = settings['filename']

    # Get the current date and time
    current_datetime = datetime.datetime.now()
    month = str(current_datetime.month)
    day = str(current_datetime.day)
    hour = str(current_datetime.hour)
    minute = str(current_datetime.minute)

    # directory_name = settings['directory name'] + f'_exp_{experiment}_{day}_{month}_{hour}_{minute}'
    # spiral_directory_name = settings['directory name'] + f'_spiral_exp_{experiment}_{day}_{month}_{hour}_{minute}'
    # quadcopter_control_direct_points(copp.sim, copp.client, vision_handle, main_route, filename, directory_name)
    #
    # copp.sim.setObjectOrientation(vision_handle, [-np.pi, np.pi / 3, -np.pi / 2], copp.sim.handle_parent)
    #
    # quadcopter_control_direct_points(copp.sim,
    #                                  copp.client,
    #                                  vision_handle,
    #                                  spiral_routes,
    #                                  'spiral_route',
    #                                  spiral_directory_name)

    spiral_route_key = spiral_route_by_target.keys()
    for route, object_key, count_group in zip(cops_route_by_group, spiral_route_key, range(len(cops_route_by_group))):
        # for route, count_group in zip(route_by_group, range(len(route_by_group))):
        filename = settings["filename"]
        vision_handle = copp.handles[settings["vision sensor names"]]

        route_of_object = cops_route_by_group[object_key]
        group_name = f"_exp_{experiment}_group_{object_key}_{day}_{month}_{hour}_{minute}"
        directory_name = settings["directory name"] + group_name

        quadcopter_control_direct_points(
            copp.sim, copp.client, vision_handle, route_of_object, filename, directory_name
        )

        # route_of_object = op_route_by_group[object_key]
        # group_name = f"_op_exp_{experiment}_group_{object_key}_{day}_{month}_{hour}_{minute}"
        # directory_name = settings["directory name"] + group_name

        # quadcopter_control_direct_points(
        #     copp.sim, copp.client, vision_handle, route_of_object, filename, directory_name
        # )

        # spiral_route = spiral_route_by_target[object_key]
        # spiral_group_name = f"_spiral_exp_{experiment}_group_{object_key}_{day}_{month}_{hour}_{minute}"
        # spiral_directory_name = settings["directory name"] + spiral_group_name

        # quadcopter_control_direct_points(
        #     copp.sim, copp.client, vision_handle, spiral_route, "spiral_route", spiral_directory_name
        # )

        # random_route = random_route_by_target[object_key]
        # random_group_name = f"_random_exp_{experiment}_group_{object_key}_{day}_{month}_{hour}_{minute}"
        # random_directory_name = settings["directory name"] + random_group_name

        # quadcopter_control_direct_points(
        #     copp.sim, copp.client, vision_handle, random_route, "random_route", random_directory_name
        # )

    with open(os.path.join(settings["save path"], f"variables/view_point_{experiment}.var"), "wb") as file:
        pickle.dump(cops_route_distace, file)
        pickle.dump(travelled_spiral_distance, file)
        pickle.dump(spiral_route_by_target, file)
        pickle.dump(cops_route_by_group, file)
        pickle.dump(spiral_target_distance, file)
        pickle.dump(random_target_distance, file)
        pickle.dump(day, file)
        pickle.dump(month, file)
        pickle.dump(hour, file)
        pickle.dump(minute, file)
