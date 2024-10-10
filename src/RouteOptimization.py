import platform
from ConvexHull.ConvexHull import convex_hull
from MathUtil.GeometryOperations import (draw_cylinder_with_hemisphere, compute_central_hemisphere_area, euler_angles_from_normal, intersect_plane_sphere, point_between_planes)
from typing import Tuple, Dict, Any, List
from numpy import ndarray, dtype, floating, float_, bool_
from numpy._typing import _64Bit
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation as Rot
from Coppelia.CoppeliaInterface import CoppeliaInterface
from random import sample
import open3d as o3d
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pyvista as pv
import os
import sys
import pickle
import shutil
import subprocess
import csv
import ast
import cv2 as cv
import datetime
import multiprocessing

from Colmap.pipeline import generate_mesh_poisson
from Colmap.colmap import run_colmap_program, statistics_colmap
from ViewPoint.ViewPoint import view_point
import config

settings = config.Settings.get()

# Variables loaded from config.yaml
CA_max = -1  # Bigger number the route has more points
max_route_radius = -1  # Bigger number the route increase the maximum points of view radius.
points_per_sphere = -1  # Density of points in the radius. If the number increase density decrease
height_proportion = -1  # The proportion of the tallest z height to make the cylinder
max_visits = -1  # Define the maximum number of times that the point can be visited
max_iter = -1  # Maximum number of iterations to try...catch a subgroup
T_max = -1  # Maximum travel budget
n_resolution = -1  # Number of subdivisions of the horizontal discretization
points_per_unit = -1
scale_to_height_spiral = 1.5  # Scale multiplied by the object target centroid Z to compute the spiral trajectory Z


def find_route(S_fr: dict):
    """
    NOT USED!!!
    Find a route based on better subgroup reward only
    :param S_fr: Dictionary with subgroups
    :return route: Subgroups that forms a route
    """
    print('Starting finding a route ...')
    route = {}
    for target, s_fr in S_fr.items():
        CA_fr_max = -1
        Si_chose = []
        for Si_fr in S_fr[target]:
            if Si_fr[-1][-1] > CA_fr_max:
                Si_chose = Si_fr
                CA_fr_max = Si_fr[-1][-1]
        route[target] = Si_chose
    return route


def get_points_to_route(route_points_gpfr: list[tuple], points_table_gpfr: list[ndarray]) -> ndarray:
    """
    Make an array with the point chose for the route. Separate it from the subgroups.
    :param route_points_gpfr: Dictionary of subgroups that forms a route
    :param points_table_gpfr: Dictionary of points of view
    :return:  Array with points
    """
    route_result_points_gpfr = np.empty(0)
    for point_pair in route_points_gpfr:
        if len(route_result_points_gpfr) == 0:
            route_result_points_gpfr = points_table_gpfr[point_pair[0]]
            continue
        route_result_points_gpfr = np.row_stack((route_result_points_gpfr, points_table_gpfr[point_pair[1]]))
    return route_result_points_gpfr


def save_points(route_sp: dict, targets_points_of_view_sr: dict):
    """
    UNDER CONSTRUCTION!!!!!
    Save the points of each route
    :param route_sp: Dictionary with subgroups with route
    :param targets_points_of_view_sr: Points of view to be converted in a route
    :return: Nothing
    """
    print('Starting saving ...')
    route_points = np.empty([0, 6])
    for target, data_s in route_sp.items():
        for data in data_s:
            point_start = targets_points_of_view_sr[target][data[0]]
            point_end = targets_points_of_view_sr[target][data[1]]
            route_points = np.row_stack((route_points, point_end))
    np.savetxt('positions.csv', route_points, delimiter=',')




def get_geometric_objects_cell(geometric_objects):
    for i in range(geometric_objects.n_cells):
        yield geometric_objects.get_cell(i)


def get_side_hemisphere_area(count_plane_gsha: int,
                             meshes_gsha: dict,
                             frustum_planes: list,
                             central_hemisphere_gsha: int) -> float:
    tmpidxs = 49 * [[]]
    number_of_elements = 0
    tmpidxs[number_of_elements] = central_hemisphere_gsha
    number_of_elements += 1
    for count_idx in range(1, 3):
        tmpidxs[number_of_elements] = (central_hemisphere_gsha + count_idx) % n_resolution + (
                central_hemisphere_gsha // n_resolution) * n_resolution
        number_of_elements += 1
        tmpidxs[number_of_elements] = (central_hemisphere_gsha - count_idx) % n_resolution + (
                central_hemisphere_gsha // n_resolution) * n_resolution
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
        ct_pt = np.array(meshes_gsha['hemispheres'][hemisphere_idx]['center'])
        is_in = False
        intersection_points = []
        for plane_gsha in frustum_planes:
            distance = (abs(np.dot(plane_gsha[:3], meshes_gsha['hemispheres'][hemisphere_idx]['center']) + plane_gsha[
                3]) / np.sqrt(plane_gsha[0] ** 2 + plane_gsha[1] ** 2 + plane_gsha[2] ** 2))
            if distance < meshes_gsha['hemispheres'][hemisphere_idx]['radius']:
                x = (-plane_gsha[3] - meshes_gsha['hemispheres'][hemisphere_idx]['center'][1] * plane_gsha[1] -
                     meshes_gsha['hemispheres'][hemisphere_idx]['center'][2] * plane_gsha[2]) / plane_gsha[0]
                point_pi = np.array([x, meshes_gsha['hemispheres'][hemisphere_idx]['center'][1],
                                     meshes_gsha['hemispheres'][hemisphere_idx]['center'][2]])
                intersection_points = intersect_plane_sphere(np.array(plane_gsha[:3]),
                                                             point_pi,
                                                             np.array(
                                                                 meshes_gsha['hemispheres'][hemisphere_idx]['center']),
                                                             meshes_gsha['hemispheres'][hemisphere_idx]['radius'])
                is_in = True
                break
        alpha = 1
        if not is_in:
            if not point_between_planes(ct_pt, np.array(frustum_planes)):
                area += 2 * alpha * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'] ** 2
            else:
                area += 0
        else:
            if point_between_planes(ct_pt, np.array(frustum_planes)):
                area += alpha * 2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'] * np.linalg.norm(
                    intersection_points[0] - intersection_points[1])
            else:
                area += alpha * (2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'] *
                                 np.linalg.norm(intersection_points[0] - intersection_points[1]) +
                                 2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'])
    return area


def get_points_route(vector_points_gpr: dict, route_gpr: dict):
    route_points = {}
    for target, data_s in route_gpr.items():
        route_points[target] = np.empty([0, 6])
        for data in data_s:
            point_start = vector_points_gpr[target][data[0]]
            point_end = vector_points_gpr[target][data[1]]
            route_points[target] = np.row_stack((route_points[target], point_end))
    return route_points


def plot_route(centroid_points_pf: dict, radius_pf: dict, target_points_pf: dict, vector_points_pr: dict):
    print('Starting showing data')
    # Create a plotter
    plotter = pv.Plotter()
    vector_points_pf = {}
    str_color = ['red', 'green', 'black']
    count_color = 0
    for target in centroid_points_pf.keys():
        cy_direction = np.array([0, 0, 1])
        n_resolution = 36
        cy_hight = height_proportion * np.max(target_points_pf[target][:, 2])
        r_mesh = radius_pf[target]
        h = np.cos(np.pi / n_resolution) * r_mesh
        l = np.sqrt(np.abs(4 * h ** 2 - 4 * r_mesh ** 2))

        # Find the radius of the spheres
        z_resolution = int(np.ceil(cy_hight / l))

        cylinder = pv.CylinderStructured(
            center=centroid_points_pf[target],
            direction=cy_direction,
            radius=r_mesh,
            height=1.0,
            theta_resolution=n_resolution,
            z_resolution=z_resolution,
        )

        points0 = vector_points_pr[target][:, :3]
        point_cloud0 = pv.PolyData(points0)
        plotter.add_mesh(point_cloud0, color=str_color[count_color])
        # arrow_direction = pv.Arrow(start=points0[0], direction=vector_points_pr[target][0, 3:])
        # plotter.add_mesh(arrow_direction, color=str_color[count_color])

        # cylinder.plot(show_edges=True)
        plotter.add_mesh(cylinder, show_edges=True)

        points = target_points_pf[target]
        point_cloud = pv.PolyData(points)
        plotter.add_mesh(point_cloud, color=str_color[count_color])
        count_color += 1

    # plotter.show()


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
            cone_name = './' + target + f'/Cone'
            handle = sim.getObject(cone_name, {'index': i, 'noError': True})
            if handle < 0:
                break
            cone_pos = list(position_orientation[i, :3])
            sim.setObjectPosition(handle, cone_pos)
        for each_position in position_orientation:
            pos = list(each_position[:3])
            next_point_handle = sim.getObject('./new_target')
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
            total_time = sim.getSimulationTime() + settings['total simulation time']
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
                diff_ori = np.subtract(orientation_angles,
                                       sim.getObjectOrientation(quad_base_handle, sim.handle_world))
                norm_diff_ori = np.linalg.norm(diff_ori)

                if norm_diff_ori > 0.08:
                    delta_ori = 0.3 * diff_ori
                    new_ori = list(sim.getObjectOrientation(quad_base_handle, sim.handle_world) + delta_ori)
                else:
                    new_ori = orientation_angles
                sim.setObjectOrientation(quad_target_handle, new_ori)
                t_stab = sim.getSimulationTime() + settings['time to stabilize']
                while sim.getSimulationTime() < t_stab:
                    diff_pos = np.subtract(new_pos,
                                           sim.getObjectPosition(quad_base_handle, sim.handle_world))
                    diff_ori = np.subtract(new_ori,
                                           sim.getObjectOrientation(quad_base_handle, sim.handle_world))
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
                print('Time short')


def compute_edge_weight_matrix(S_cewm: dict, targets_points_of_view_cewm: dict[Any, ndarray]) -> ndarray:
    print('Starting computing distance matrix')

    i = 0
    j = 0
    total_length = 0
    for _, points_start_cewm in targets_points_of_view_cewm.items():
        total_length += points_start_cewm.shape[0]

    edge_weight_matrix_cewm = np.zeros([total_length, total_length])
    for _, points_start_cewm in targets_points_of_view_cewm.items():
        for pt1 in points_start_cewm:
            for _, points_end_cewm in targets_points_of_view_cewm.items():
                for pt2 in points_end_cewm:
                    edge_weight_matrix_cewm[i, j] = np.linalg.norm(pt1[:3] - pt2[:3]) + np.linalg.norm(
                        np.deg2rad(pt1[3:]) - np.deg2rad(pt2[3:]))
                    j += 1
            i += 1
            j = 0

    # edge_weight_matrix_cewm = np.zeros([length_start,length_start])
    #
    # for target_cewm, S_cewm_start in S_cewm.items():
    #     # if i == 0 and j == 0:
    #     #     edge_weight_matrix_cewm = np.zeros([2 * ((len(S_cewm_start) - 1) * len(settings['object names'])) - 1,
    #     #                                         2 * ((len(S_cewm_start) - 1) * len(settings['object names'])) - 1])
    #     for Si_cewm_start in S_cewm_start:
    #         j = 0
    #         # count_target_i = 0
    #         if Si_cewm_start[-1][6] > length_start + len(S_cewm_start) or Si_cewm_start[-1][7] > length_start + len(S_cewm_start):
    #             print('There are something wrong')
    #         length_end = 0
    #         idx1 = Si_cewm_start[-1][1]  # - count_target*targets_points_of_view_cewm[target_cewm].shape[0]
    #         conversion_table = 7000 * [[]]
    #         for target_cewm_i, S_cewm_end in S_cewm.items():
    #             for Si_cewm_end in S_cewm_end:
    #                 idx2 = Si_cewm_end[0][1]  # - count_target_i*targets_points_of_view_cewm[target_cewm_i].shape[0]
    #                 pt1 = targets_points_of_view_cewm[target_cewm][idx1]
    #                 pt2 = targets_points_of_view_cewm[target_cewm_i][idx2]
    #                 if i != j:
    #                     edge_weight_matrix_cewm[Si_cewm_start[-1][6], Si_cewm_end[0][7]] = np.linalg.norm(pt1 - pt2)
    #                 conversion_table[j] = [Si_cewm_end[0][0], j]
    #                 j += 1
    #             length_end += len(S_cewm_end)
    #         i += 1
    #     length_start += len(S_cewm_start)
    # i -= 1
    # j -= 1
    # edge_weight_matrix_cewm = edge_weight_matrix_cewm[:i, :j]
    print(f'size of edge matrix: {edge_weight_matrix_cewm.shape[0]} x {edge_weight_matrix_cewm.shape[0]}')
    return edge_weight_matrix_cewm


def read_problem_file(filename: str) -> dict:
    read_fields = {}
    line_count = 0
    try:
        with open(filename, 'r') as file:
            for line in file:
                # Split the line using ":" as the delimiter
                parts = line.strip().split(':')
                # Ensure there are two parts after splitting
                if len(parts) == 2:
                    if parts[1].isdigit():
                        read_fields[parts[0]] = int(parts[1])
                    else:
                        read_fields[parts[0]] = parts[1]
                else:
                    read_fields[f'{line_count}'] = parts[0]
    except FileNotFoundError:
        print("File not found:", filename)
    except Exception as e:
        print("An error occurred:", e)
    return read_fields


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
    print('Generating spiral trajectory over one target')
    step = radius_gstst / parts_gstst
    points_gstst = generate_spiral_points(radius_gstst, step)

    points_gstst = np.array([np.hstack((np.array(p), 0)) for p in points_gstst])
    directions_gstst = np.zeros([1, 3])
    for p in points_gstst[1:]:
        directions_gstst = np.row_stack((directions_gstst, euler_angles_from_normal(-p)))
    points_gstst = points_gstst + centroid_points_gstst

    return points_gstst, directions_gstst


def remove_unused_files(workspace_folder):
    """
    Remove the stereo folder and the images folder in the dense folder
    """
    dense_folder = os.path.join(workspace_folder, 'dense')
    for folder in os.listdir(dense_folder):
        stereo_folder = os.path.join(folder, 'stereo')
        images_folder = os.path.join(folder, 'images')
        vis_file = os.path.join(folder, 'fused.ply.vis')

        stereo_folder_path = os.path.join(dense_folder, stereo_folder)
        images_folder_path = os.path.join(dense_folder, images_folder)
        vis_file_path = os.path.join(dense_folder, vis_file)

        shutil.rmtree(stereo_folder_path)
        shutil.rmtree(images_folder_path)
        os.remove(vis_file_path)


def point_cloud(experiment: int) -> None:
    with open(os.path.join(settings['save path'], f'variables/view_point_{experiment}.var'), 'rb') as f:
        travelled_distance_main = pickle.load(f)
        travelled_spiral_distance = pickle.load(f)
        spiral_route_by_target = pickle.load(f)
        route_by_group = pickle.load(f)
        spiral_target_distance = pickle.load(f)
        random_target_distance = pickle.load(f)
        day = pickle.load(f)
        month = pickle.load(f)
        hour = pickle.load(f)
        minute = pickle.load(f)

    # Get the current date and time
    workspace_folder = os.path.join(settings['workspace folder'], f'exp_{experiment}_{day}_{month}_{hour}_{minute}')
    spiral_workspace_folder = os.path.join(settings['workspace folder'],
                                           f'spiral_exp_{experiment}_{day}_{month}_{hour}_{minute}')

    # directory_name = settings['directory name'] + f'_exp_{experiment}_{day}_{month}_{hour}_{minute}'
    spiral_directory_name = settings['directory name'] + f'_spiral_exp_{experiment}_{day}_{month}_{hour}_{minute}'

    colmap_folder = settings['colmap folder']

    # remove folder if exist
    if os.path.exists(workspace_folder):
        shutil.rmtree(workspace_folder)

    # Create the directory
    os.makedirs(workspace_folder)

    with open(os.path.join(workspace_folder, 'distance.txt'), 'w') as distance_file:
        distance_file.write(f'distance: {str(travelled_distance_main)}\n')
        distance_file.write(f'CA_max: {settings["CA_max"]}\n')
        distance_file.write(f'CA_min: {settings["CA_min"]}')
    
    # images_folder = str(os.path.join(settings['path'], directory_name))
    # run_colmap_program(colmap_folder, workspace_folder, images_folder)
    # statistics_colmap(colmap_folder, workspace_folder)

    # remove folder if exist
    if os.path.exists(spiral_workspace_folder):
        shutil.rmtree(spiral_workspace_folder)

    # Create the directory
    # os.makedirs(spiral_workspace_folder)

    # with open(spiral_workspace_folder + '/distance.txt', 'w') as distance_file:
    #     distance_file.write(str(travelled_spiral_distance))

    # spiral_images_folder = str(os.path.join(settings['path'], spiral_directory_name))
    # run_colmap_program(colmap_folder, spiral_workspace_folder, spiral_images_folder)
    # statistics_colmap(colmap_folder, spiral_workspace_folder)

    MNRE_array = np.empty(0)
    spiral_route_key = spiral_route_by_target.keys()
    for route, object_key, count_group in zip(route_by_group, spiral_route_key, range(len(route_by_group))):
        # for route, count_group in zip(route_by_group, range(len(route_by_group))):

        # Recosntrution COPS
        workspace_folder = os.path.join(settings['workspace folder'],
                                        f'exp_{experiment}_{day}_{month}_{hour}_{minute}_group_{object_key}')
        
        image_directory_name = (settings['directory name'] +
                                f'_exp_{experiment}_group_{object_key}_{day}_{month}_{hour}_{minute}')

        # remove folder if exist
        if os.path.exists(workspace_folder):
            shutil.rmtree(workspace_folder)

        # Create the directory
        os.makedirs(workspace_folder)

        travelled_distance_main = 0
        route_points = route_by_group[object_key]
        for i in range(1, len(route_by_group[object_key])):
            travelled_distance_main += np.linalg.norm(route_points[i - 1][:3] - route_points[i][:3])

        with open(os.path.join(workspace_folder, 'distance.txt'), 'w') as distance_file:
            distance_file.write(str(travelled_distance_main))

        with open(os.path.join(workspace_folder, 'object_name.txt'), 'w') as object_name_file:
            object_name_file.write(object_key)

        images_folder = str(os.path.join(settings['path'], image_directory_name))
        # run_colmap_program(colmap_folder, workspace_folder, images_folder)
        # MNRE_array = statistics_colmap(colmap_folder, workspace_folder, MNRE_array)
        # remove_unused_files(workspace_folder)

        # Recosntrution OP
        op_workspace_folder = os.path.join(settings['workspace folder'],
                                        f'op_exp_{experiment}_{day}_{month}_{hour}_{minute}_group_{object_key}')
        
        op_image_directory_name = (settings['directory name'] +
                                f'_op_exp_{experiment}_group_{object_key}_{day}_{month}_{hour}_{minute}')

        # remove folder if exist
        if os.path.exists(op_workspace_folder):
            shutil.rmtree(op_workspace_folder)

        # Create the directory
        os.makedirs(op_workspace_folder)

        travelled_distance_main = 0
        route_points = route_by_group[object_key]
        for i in range(1, len(route_by_group[object_key])):
            travelled_distance_main += np.linalg.norm(route_points[i - 1][:3] - route_points[i][:3])

        with open(os.path.join(op_workspace_folder, 'distance.txt'), 'w') as distance_file:
            distance_file.write(str(travelled_distance_main))

        with open(os.path.join(op_workspace_folder, 'object_name.txt'), 'w') as object_name_file:
            object_name_file.write(object_key)

        images_folder = str(os.path.join(settings['path'], op_image_directory_name))
        # run_colmap_program(colmap_folder, op_workspace_folder, images_folder)
        # MNRE_array = statistics_colmap(colmap_folder, op_workspace_folder, MNRE_array)
        # remove_unused_files(op_workspace_folder)

        # Recosntrution Spiral
        spiral_workspace_folder = os.path.join(settings['workspace folder'],
                                               f'spiral_exp_{experiment}_{day}_{month}_{hour}_{minute}_group_{object_key}')

        spiral_directory_name = settings[
                                    'directory name'] + f'_spiral_exp_{experiment}_group_{object_key}_{day}_{month}_{hour}_{minute}'

        # remove folder if exist
        if os.path.exists(spiral_workspace_folder):
            shutil.rmtree(spiral_workspace_folder)

        # Create the directory
        os.makedirs(spiral_workspace_folder)

        with open(os.path.join(spiral_workspace_folder, 'distance.txt'), 'w') as distance_file:
            distance_file.write(str(spiral_target_distance[object_key]))

        with open(os.path.join(spiral_workspace_folder, 'object_name.txt'), 'w') as object_name_file:
            object_name_file.write(object_key)

        spiral_images_folder = str(os.path.join(settings['path'], spiral_directory_name))
        # run_colmap_program(colmap_folder, spiral_workspace_folder, spiral_images_folder)
        # statistics_colmap(colmap_folder, spiral_workspace_folder)
        # remove_unused_files(spiral_workspace_folder)

        # Recosntrution Random
        random_workspace_folder = os.path.join(settings['workspace folder'],
                                               f'random_exp_{experiment}_{day}_{month}_{hour}_{minute}_group_{object_key}')

        random_directory_name = settings[
                                    'directory name'] + f'_random_exp_{experiment}_group_{object_key}_{day}_{month}_{hour}_{minute}'

        # remove folder if exist
        if os.path.exists(random_workspace_folder):
            shutil.rmtree(random_workspace_folder)

        # Create the directory
        os.makedirs(random_workspace_folder)

        with open(os.path.join(random_workspace_folder, 'distance.txt'), 'w') as distance_file:
            distance_file.write(str(random_target_distance[object_key]))

        with open(os.path.join(random_workspace_folder, 'object_name.txt'), 'w') as object_name_file:
            object_name_file.write(object_key)

        random_images_folder = str(os.path.join(settings['path'], random_directory_name))
        run_colmap_program(colmap_folder, random_workspace_folder, random_images_folder)
        statistics_colmap(colmap_folder, random_workspace_folder)
        remove_unused_files(random_workspace_folder)


def read_camera_pos_files(file_path: str, ref_file_path: str) -> dict:
    """Read the camera position files and return a dictionary with the camera positions"""

    # Ler o arquivo e extrair apenas os centros das câmeras
    camera_centers = {}

    with open(file_path, "r") as f:
        # Ignorar as primeiras quatro linhas
        for _ in range(4):
            f.readline()

        line = f.readline()
        while line:
            line_split = line.replace("\n", "").split(" ")
            image_name = line_split[-1]
            qw, qx, qy, qz = map(float, line_split[1:5])
            tx, ty, tz = map(float, line_split[5:8])

            # Converter quaternion para matriz de rotação
            rotation = Rot.from_quat([qx, qy, qz, qw])
            R = rotation.as_matrix()

            # Calcular o centro da câmera
            t = np.array([tx, ty, tz])
            camera_center = -R.T @ t
            camera_centers[image_name] = [camera_center]

            # Ler a linha vazia e a próxima linha com dados
            f.readline()
            line = f.readline()

    with open(ref_file_path, "r") as f:
        line = f.readline()
        while line:
            line_split = line.replace("\n", "").split(" ")
            image_name = line_split[0]

            real_camera_center = list(map(float, line_split[1:4]))

            if image_name in camera_centers:
                camera_centers[image_name].append(np.array(real_camera_center))

            line = f.readline()

    return camera_centers


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
    scale = np.sum(S) / np.sum(P_centered ** 2)

    # Compute the translation
    t = centroid_Q - scale * np.dot(R, centroid_P)

    # Create the homogeneous transformation matrix
    T = np.identity(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t

    return T


def align_points(source_cloud, target_cloud):
    """Align two point clouds using Horn's method"""
    source_points = np.array(source_cloud.points)
    target_points = np.array(target_cloud.points)

    return horn(source_points, target_points)


def metrics(source_points: ndarray, target_points: ndarray) -> dict:
    """Calculate the metrics between two point clouds"""
    n = len(source_points)
    min = np.inf
    max = -np.inf
    mae = 0
    rmse = 0

    if len(target_points) == 0 or n == 0:
        return {
            "source_points": n,
            "min": min,
            "max": max,
            "mae": -np.inf,
            "rmse": -np.inf
        }

    for point in source_points:
        dist = np.linalg.norm(target_points - point, axis=1)
        curr_min = np.min(dist)

        if curr_min < min:
            min = curr_min

        if curr_min > max:
            max = curr_min

        mae += curr_min
        rmse += curr_min ** 2

    mae /= n
    rmse = np.sqrt(rmse / n)

    return {
        "source_points": n,
        "min": min,
        "max": max,
        "mae": mae,
        "rmse": rmse
    }


def calculate_metrics_distance(source_pcd, target_pcd):
    """Calculate the metrics between two point clouds"""
    source = np.array(source_pcd.points)
    target = np.array(target_pcd.points)

    source = source[~np.isnan(source).any(axis=1)]
    target = target[~np.isnan(target).any(axis=1)]

    print(f"Metrics PointCloud with {len(source)} points")
    print(f"Metrics PointCloud with {len(target)} points")

    params = [(source, target), (target, source)]
    return Pool().starmap(metrics, params)


def get_last_directories(path, count):
    """Return the last directories of a path"""
    path_parts = path.split(os.sep)

    if path_parts[-1] == '':
        path_parts.pop()

    last_three_dirs = os.path.join(*path_parts[-count:])

    return last_three_dirs


def back_directories(path, count):
    """Navigate back through the last directories of a path"""
    path_parts = path.split(os.sep)

    if path_parts[-1] == '':
        path_parts.pop()

    n = len(path_parts)

    if path_parts[0] == '':
        path_parts[0] = os.sep

    last_three_dirs = os.path.join(*path_parts[:n - count])

    return last_three_dirs


def save_to_csv(data, filename):
    """Save the data to a CSV file"""
    keys = data[0].keys()
    with open(filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()

        for line in data:
            if line is None:
                continue

            dict_writer.writerow(line)


def get_tranformation_matrix(camera_cloud, real_camera_cloud):
    """Return the transformation matrix between two point clouds"""
    transformation = align_points(camera_cloud, real_camera_cloud)
    camera_cloud.transform(transformation)

    cam_error = np.linalg.norm(np.array(camera_cloud.points) - np.array(real_camera_cloud.points)) / len(
        np.array(camera_cloud.points)
    )

    T = transformation
    while cam_error > 0.0005:
        n = len(camera_cloud.points)
        colmap_points = np.array(camera_cloud.points)
        real_points = np.array(real_camera_cloud.points)

        max_cam_error_find = 0
        k = -1
        min = np.inf
        min_camera_error = 0.2
        for i in range(n):
            camera_error = np.linalg.norm(colmap_points[i] - real_points[i])
            if camera_error > max_cam_error_find and camera_error > min_camera_error:
                max_cam_error_find = camera_error
                k = i

            if camera_error < min:
                min = camera_error

        if not n > 2:
            raise Exception("Can not find the transformation matrix")

        camera_cloud.transform(np.linalg.inv(T))
        colmap_points = np.array(camera_cloud.points)
        real_points = np.array(real_camera_cloud.points)

        if k == -1:
            break

        colmap_points = np.delete(colmap_points, k, 0)
        real_points = np.delete(real_points, k, 0)

        camera_cloud.points = o3d.utility.Vector3dVector(colmap_points)
        real_camera_cloud.points = o3d.utility.Vector3dVector(real_points)

        T = horn(colmap_points, real_points)

        camera_cloud.transform(T)

        cam_error = np.linalg.norm(np.array(colmap_points) - np.array(real_points)) / n
        print(f"{cam_error=:.8f}")

    return T


def crop_point_cloud(target_pcd, bounding_box):
    """Crop a point cloud according to the bounding box"""
    bounding_box_points = np.array(bounding_box.get_box_points())
    b_min = np.min(bounding_box_points, axis=0)
    b_max = np.max(bounding_box_points, axis=0)
    bounding_box_points_dist = np.linalg.norm(b_min - b_max)
    # threshold = bounding_box_points_dist * 0.01
    threshold = 0.00
    print(f"{b_min=}")
    print(f"{b_max=}")
    print(f"{bounding_box_points_dist=}")
    print(f"{threshold=}")

    target_pcd_points = np.array(target_pcd.points)

    bounding_points = []
    for p in target_pcd_points:
        if p[0] < b_min[0] - threshold or p[1] < b_min[1] - threshold or p[2] < b_min[2] - threshold:
            continue
        elif p[0] > b_max[0] + threshold or p[1] > b_max[1] + threshold or p[2] > b_max[2] + threshold:
            continue
        else:
            bounding_points.append(p)

    return bounding_points


def process_reconstruction(image_path, reconstruction_path, plt_path):
    """Process the reconstruction and calculate the metrics"""
    sparce_folder = "sparse"
    file_name = "images.txt"
    ref_file_name = "ref_images.txt"
    mesh_file = "meshed-poisson.ply"

    sparce_path = os.path.join(reconstruction_path, sparce_folder)
    file_path = os.path.join(sparce_path, file_name)
    ref_file_path = os.path.join(image_path, ref_file_name)
    mesh_plt_path = os.path.join(reconstruction_path, mesh_file)

    camera_centers = read_camera_pos_files(file_path, ref_file_path)

    colmap_camera_pos = []
    real_camera_pos = []

    for pos in camera_centers.values():
        colmap_camera_pos.append(pos[0])
        real_camera_pos.append(pos[1])

    camera_cloud = o3d.geometry.PointCloud()
    camera_cloud.points = o3d.utility.Vector3dVector(colmap_camera_pos)

    real_camera_cloud = o3d.geometry.PointCloud()
    real_camera_cloud.points = o3d.utility.Vector3dVector(real_camera_pos)

    try:
        T = get_tranformation_matrix(camera_cloud, real_camera_cloud)
    except Exception as e:
        print(e)
        return None

    print("Used Transformation Matrix:")
    print(T)

    target_mesh = o3d.io.read_triangle_mesh(mesh_plt_path)
    target_mesh.transform(T)

    source_pcd = o3d.io.read_point_cloud(plt_path)
    target_pcd = o3d.io.read_point_cloud(mesh_plt_path)

    target_pcd.transform(T)

    bbp = source_pcd.get_axis_aligned_bounding_box()
    bounding_points = crop_point_cloud(target_pcd, bbp)

    cropped_mesh = target_mesh.crop(bbp)

    target_pcd.points = o3d.utility.Vector3dVector(bounding_points)

    new_mesh_path = os.path.dirname(mesh_plt_path)

    o3d.io.write_point_cloud(os.path.join(new_mesh_path, 'point-cloud-crop.ply'), target_pcd)
    o3d.io.write_triangle_mesh(os.path.join(new_mesh_path, 'meshed-poisson-crop.ply'), cropped_mesh)

    print("Start calculate hausdorff_distance")
    metrics_dist = calculate_metrics_distance(source_pcd, target_pcd)
    last_dir = get_last_directories(reconstruction_path, 3)
    print(last_dir)

    distance_file_path = back_directories(reconstruction_path, 2)
    distance_file_path = os.path.join(distance_file_path, 'distance.txt')

    with open(distance_file_path, 'r') as file:
        dist = float(file.readline().strip())

    experiment = get_experiment_number(reconstruction_path)

    if last_dir.startswith('spiral') or last_dir.startswith('op') or last_dir.startswith('random'):
        route_reward = 'none'

    else:
        results_cops_file = f"{settings['COPS problem']}{experiment}.csv"
        with open(os.path.join(settings['COPS result'], results_cops_file),'r') as csv_reward_file:
            csv_reader = csv.DictReader(csv_reward_file, delimiter=';')

            line = next(csv_reader)
            route_reward = line['profit']
            route_reward = float(route_reward.replace(',', '.'))

    return {
        'reconstruction_path': last_dir,
        'ply': os.path.basename(plt_path),
        'source_points': metrics_dist[0]['source_points'],
        'distance': dist,
        'min': np.min((metrics_dist[0]['min'], metrics_dist[1]['min'])),
        'max': np.max((metrics_dist[0]['max'], metrics_dist[1]['max'])),
        'mae': metrics_dist[0]['mae'] + metrics_dist[1]['mae'],
        'rmse': metrics_dist[0]['rmse'] + metrics_dist[1]['rmse'],
        'route_reward': route_reward,
    }


def get_experiment_number(reconstruction_path: str):
    path = back_directories(reconstruction_path, 2)
    path = get_last_directories(path, 1)

    split_name = path.split('_')

    if split_name[0] == 'spiral':
        return split_name[2]
    
    return split_name[1]


def process_paths(list_image_path, list_reconstruction_path, list_plt_path):
    data = []

    for image_path, reconstruction_path, plt_path in zip(list_image_path, list_reconstruction_path, list_plt_path):
        data.append(process_reconstruction(image_path, reconstruction_path, plt_path))

    save_to_csv(data, os.path.join(settings['save path'], 'metrics.csv'))


def mesh_analysis():
    print('Initiating mesh analysis')

    obj_name = set()
    workspace_folder = settings['workspace folder']
    for folder in os.listdir(workspace_folder):
        object_name_file = os.path.join(settings['workspace folder'], folder)
        object_name_file = os.path.join(object_name_file, 'object_name.txt')

        if not os.path.isfile(object_name_file):
            continue

        with open(object_name_file, 'r') as file:
            obj = file.readline().strip()
            obj_name.add(obj)

    list_image_path = []
    list_reconstruction_path = []
    list_plt_path = []

    experiment = settings['number of trials']
    for exp in range(experiment):
        with open(os.path.join(settings['save path'], f'variables/view_point_{exp}.var'), 'rb') as f:
            travelled_distance_main = pickle.load(f)
            travelled_spiral_distance = pickle.load(f)
            spiral_route_by_target = pickle.load(f)
            route_by_group = pickle.load(f)
            spiral_target_distance = pickle.load(f)
            random_target_distance = pickle.load(f)
            day = pickle.load(f)
            month = pickle.load(f)
            hour = pickle.load(f)
            minute = pickle.load(f)

        for obj in obj_name:
            workspace_folder_group = os.path.join(settings['workspace folder'], f'exp_{exp}_{day}_{month}_{hour}_{minute}_group_{obj}')
            op_workspace_folder_group = os.path.join(settings['workspace folder'], f'op_exp_{exp}_{day}_{month}_{hour}_{minute}_group_{obj}')
            spiral_workspace_folder_group = os.path.join(settings['workspace folder'], f'spiral_exp_{exp}_{day}_{month}_{hour}_{minute}_group_{obj}')
            random_workspace_folder_group = os.path.join(settings['workspace folder'], f'random_exp_{exp}_{day}_{month}_{hour}_{minute}_group_{obj}')

            images_folder = os.path.join(settings['path'], f'scene_builds_exp_{exp}_group_{obj}_{day}_{month}_{hour}_{minute}')
            images_folder_op = os.path.join(settings['path'], f'scene_builds_op_exp_{exp}_group_{obj}_{day}_{month}_{hour}_{minute}')
            images_folder_spiral = os.path.join(settings['path'], f'scene_builds_spiral_exp_{exp}_group_{obj}_{day}_{month}_{hour}_{minute}')
            images_folder_random = os.path.join(settings['path'], f'scene_builds_random_exp_{exp}_group_{obj}_{day}_{month}_{hour}_{minute}')

            ply_path = f'mesh_obj/{obj}.ply'
            if os.path.isdir(workspace_folder_group):

                dense_folder = os.path.join(workspace_folder_group, 'dense')

                if os.path.isdir(dense_folder):
                    for i in os.listdir(dense_folder):
                        reconstruction_path = os.path.join(dense_folder, i)

                        list_image_path.append(images_folder)
                        list_reconstruction_path.append(reconstruction_path)
                        list_plt_path.append(ply_path)

            if os.path.isdir(op_workspace_folder_group):

                dense_folder = os.path.join(op_workspace_folder_group, 'dense')
                if os.path.isdir(dense_folder):
                    for i in os.listdir(dense_folder):
                        reconstruction_path = os.path.join(dense_folder, i)

                        list_image_path.append(images_folder_op)
                        list_reconstruction_path.append(reconstruction_path)
                        list_plt_path.append(ply_path)

            if os.path.isdir(spiral_workspace_folder_group):

                dense_folder = os.path.join(spiral_workspace_folder_group, 'dense')
                if os.path.isdir(dense_folder):
                    for i in os.listdir(dense_folder):
                        reconstruction_path = os.path.join(dense_folder, i)

                        list_image_path.append(images_folder_spiral)
                        list_reconstruction_path.append(reconstruction_path)
                        list_plt_path.append(ply_path)

            if os.path.isdir(random_workspace_folder_group):

                dense_folder = os.path.join(random_workspace_folder_group, 'dense')
                if os.path.isdir(dense_folder):
                    for i in os.listdir(dense_folder):
                        reconstruction_path = os.path.join(dense_folder, i)

                        list_image_path.append(images_folder_random)
                        list_reconstruction_path.append(reconstruction_path)
                        list_plt_path.append(ply_path)

    process_paths(list_image_path, list_reconstruction_path, list_plt_path)


def update_current_experiment(value_stage: int) -> None:
    with open(os.path.join(settings['save path'], '.progress'), 'wb') as file:
        pickle.dump(value_stage, file)


def generate_poisson_mesh() -> None:
    if platform.system() == 'Windows':
        colmap_exec = os.path.join(settings['colmap folder'], 'COLMAP.bat')

    if platform.system() == 'Linux':
        colmap_exec = 'colmap'

    save_path = settings['workspace folder']
    for workspace_folder in os.listdir(save_path):

        folder_path = os.path.join(save_path, workspace_folder)
        folder_path = os.path.join(folder_path, 'dense')

        # Check if the 'dense' folder exists
        if not os.path.isdir(folder_path):
            continue

        for folder_i in os.listdir(folder_path):
            print(os.path.join(workspace_folder, folder_i))
            curr_dir = os.path.join(folder_path, folder_i)

            # Check if 'meshed-poisson.ply' file exist
            if os.path.isfile(os.path.join(curr_dir, 'meshed-poisson.ply')):
                continue

            # Check if 'fused.ply' file does not exist
            if not os.path.isfile(os.path.join(curr_dir, 'fused.ply')):
                print(f"Cannot find 'fused.ply' in {curr_dir}")
                continue

            print(colmap_exec, workspace_folder, curr_dir)
            generate_mesh_poisson(colmap_exec, os.path.join(save_path, workspace_folder), curr_dir)


def execute_experiment() -> None:
    # Create the directory
    os.makedirs(os.path.join(settings['save path'], 'variables'), exist_ok=True)

    with open(os.path.join(settings['save path'], '.progress'), 'rb') as f:
        last_expe = pickle.load(f)

    try:
        if len(sys.argv) < 2:

            for experiment in range(settings['number of trials']):
                if experiment < last_expe:
                    continue

                proc = multiprocessing.Process(target=convex_hull, args=(experiment,))
                proc.start()
                proc.join()

                copp = CoppeliaInterface(settings)
                view_point(copp, experiment)

                copp.sim.stopSimulation()
                del copp

                point_cloud(experiment)

                generate_poisson_mesh(experiment)

                mesh_analysis()
                update_current_experiment(int(experiment + 1))

            os.remove(os.path.join(settings['save path'], '.progress'))
            return

        if sys.argv[1] == 'convex_hull':
            for experiment in range(settings['number of trials']):
                if experiment < last_expe:
                    continue

                proc = multiprocessing.Process(target=convex_hull, args=(experiment,))
                proc.start()
                proc.join()
                update_current_experiment(int(experiment + 1))

            os.remove(os.path.join(settings['save path'], '.progress'))
            return

        if sys.argv[1] == 'view_point':
            copp = CoppeliaInterface(settings)
            for experiment in range(settings['number of trials']):
                if experiment < last_expe:
                    continue

                view_point(copp, experiment)
                update_current_experiment(int(experiment + 1))

            os.remove(os.path.join(settings['save path'], '.progress'))
            copp.sim.stopSimulation()
            return

        if sys.argv[1] == 'point_cloud':
            for experiment in range(settings['number of trials']):
                if experiment < last_expe:
                    continue

                point_cloud(experiment)
                update_current_experiment(int(experiment + 1))

            os.remove(os.path.join(settings['save path'], '.progress'))
            return

        if sys.argv[1] == 'poisson_check':
            generate_poisson_mesh()
            return

        if sys.argv[1] == 'mesh_analysis':
            mesh_analysis()
            return

    except RuntimeError as e:
        print("An error occurred:", e)


def load_variables():

    if len(sys.argv) >= 7:
        settings['points per unit'] = float(sys.argv[2])
        settings['T_max'] = int(sys.argv[3])
        settings['CA_min'] = int(sys.argv[4])
        settings['CA_max'] = int(sys.argv[5])
        settings['obj_file'] = sys.argv[6]

    global CA_max
    CA_max = float(settings['CA_max'])
    global max_route_radius
    max_route_radius = float(settings['max route radius'])
    global points_per_sphere
    points_per_sphere = float(settings['points per sphere'])
    global height_proportion
    height_proportion = float(settings['height proportion'])
    global max_visits
    max_visits = int(settings['max visits'])
    global max_iter
    max_iter = int(settings['max iter'])
    global T_max
    T_max = float(settings['T_max'])
    global n_resolution
    n_resolution = int(settings['n resolution'])
    global points_per_unit
    points_per_unit = float(settings['points per unit'])

    settings['save path'] = os.path.abspath(settings['save path'])

    save_path = settings['save path']

    path = settings['path']
    COPS_dataset = settings['COPS dataset']
    COPS_result = settings['COPS result']
    workspace_folder = settings['workspace folder']

    settings['path'] = os.path.join(save_path, path)
    settings['COPS dataset'] = os.path.join(save_path, COPS_dataset)
    settings['COPS result'] = os.path.join(save_path, COPS_result)
    settings['workspace folder'] = os.path.join(save_path, workspace_folder)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    load_variables()

    save_path = settings['save path']

    os.makedirs(save_path, exist_ok=True)

    # check if file not exits
    progress_file = os.path.join(settings['save path'], '.progress')
    if not os.path.isfile(progress_file):
        with open(progress_file, 'wb') as file:
            pickle.dump(0, file)

    execute_experiment()