import os
from typing import Tuple, Dict, Any, List

import numpy as np
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit
from scipy.spatial import ConvexHull, Delaunay
from CoppeliaInterface import CoppeliaInterface
import pyvista as pv
from config import parse_settings_file
from random import sample
from GeometryOperations import (draw_cylinder_with_hemisphere,
                                compute_central_hemisphere_area,
                                get_point_intersection_plane_with_sphere, calculate_spherical_side_area,
                                intersect_plane_sphere)
import shutil
import subprocess
import csv
import ast
import cv2 as cv
import datetime

# Variables loaded from config.yaml
CA_max = -1  # Bigger number the route has more points
max_route_radius = -1  # Bigger number the route increase the maximum radius of the points of view.
points_per_sphere = -1  # Density of points in the radius. If the number increase density decrease
height_proportion = -1  # The proportion of the tallest z height to make the cylinder
max_visits = -1  # Define the maximum number of times that the point can be visited
max_iter = -1  # Maximum number of iteration to try catch a subgroup
T_max = -1  # Maximum travel budget
n_resolution = -1  # Number of subdivision of the horizontal discretization
points_per_unit = -1

global settings


def run_colmap(colmap_folder: str, workspace_folder: str, image_folder: str) -> None:
    try:
        # Execute the script using subprocess
        process = subprocess.Popen([colmap_folder + 'COLMAP.bat',
                                    'automatic_reconstructor',
                                    '--image_path',
                                    image_folder,
                                    '--workspace_path',
                                    workspace_folder])

        # Wait for the process to finish
        stdout, stderr = process.communicate()

        # Check if there were any errors
        if process.returncode != 0:
            print("Error executing script:")
            print(stderr.decode('utf-8'))
        else:
            print("Script executed successfully.")
    except Exception as e:
        print("An error occurred:", e)


def camera_view_evaluation(targets_points_of_view_cve: dict):
    print('Starting creating evaluation matrix')
    points_of_view_contribution = {}
    for target, points in targets_points_of_view_cve.items():
        points_of_view_contribution[target] = 20 * np.random.rand(points.shape[0])
    return points_of_view_contribution


def is_point_inside(point, hull):
    # Check if the given point is within the convex hull
    point_in_hull = hull.find_simplex(point) >= 0
    return point_in_hull


def is_line_through_convex_hull(hull, line):
    for point in line:
        if is_point_inside(point, hull):
            return True
    return False


def points_along_line(start_point, end_point, num_points):
    # Generate num_points equally spaced between start_point and end_point
    x = np.linspace(start_point[0], end_point[0], num_points)
    y = np.linspace(start_point[1], end_point[1], num_points)
    z = np.linspace(start_point[2], end_point[2], num_points)
    points = np.column_stack((x, y, z))
    return points


def subgroup_formation(targets_border_sf: dict, points_of_view_contribution_sf: dict,
                       target_points_of_view_sf: dict, positions_sf: dict) -> dict:
    print('Starting subgroup formation')
    S = {}
    contribution = 0
    # plotter_sf = pv.Plotter()
    subgroup_idx = 0
    is_first_target = True
    cont_target = 0
    length = 0
    for target, points in target_points_of_view_sf.items():
        S[target] = []
        if subgroup_idx == 0:
            S[target].append([])
            S[target][-1].append((subgroup_idx, subgroup_idx, 0, 0, 0.0, 0.0, 0, 0))
            subgroup_idx += 1
        visits_to_position = np.zeros(points.shape[0])
        if is_first_target:
            visits_to_position[0] += max_visits + 1
        indexes_of_ini_points = list(range(points.shape[0]))
        random_points = sample(indexes_of_ini_points, len(indexes_of_ini_points))
        show_number_of_points = 0
        for i in random_points:
            CA = 0
            total_distance = 0
            S[target].append([])
            prior_idx = i
            max_idx = -1
            idx_list = [i]
            # print(f'Point {show_number_of_points} of {len(indexes_of_ini_points)}')
            show_number_of_points += 1
            iteration = 0
            while CA < CA_max and iteration < max_iter:
                iteration += 1
                indexes_of_points = np.random.randint(low=0, high=points.shape[0], size=20)
                max_contribution = 0
                for index in indexes_of_points:
                    if index in idx_list:
                        continue
                    is_point_inside_sf = False
                    is_line_through_convex_hull_sf = False
                    for target_compare, hull_sf in targets_border_sf.items():

                        is_point_inside_sf = is_point_inside(points[index, :3], hull_sf)
                        if is_point_inside_sf:
                            break
                        line_points = points_along_line(target_points_of_view_sf[target][prior_idx, :3],
                                                        target_points_of_view_sf[target][index, :3], 20)
                        # point_cloud = pv.PolyData(positions_sf[target_compare])
                        # plotter_sf.add_mesh(point_cloud)
                        # points_line = pv.PolyData(line_points)
                        # plotter_sf.add_mesh(points_line, color='red')
                        # pl_sf = pv.Plotter()
                        # meshes = plotter_sf.meshes
                        # for mesh in meshes:
                        #     pl_sf.add_mesh(mesh)
                        # pl_sf.show()
                        is_line_through_convex_hull_sf = is_line_through_convex_hull(hull_sf, line_points)
                        if is_line_through_convex_hull_sf:
                            break
                    if is_point_inside_sf:
                        continue
                    if is_line_through_convex_hull_sf:
                        continue
                    distance_p2p = np.linalg.norm(
                        target_points_of_view_sf[target][prior_idx, :3] - target_points_of_view_sf[target][index, :3])
                    contribution = abs(abs(points_of_view_contribution_sf[target][index]) - distance_p2p)
                    if contribution > max_contribution:
                        max_idx = index
                        max_contribution = abs(points_of_view_contribution_sf[target][index])

                if max_idx == -1:
                    continue

                if visits_to_position[max_idx] > max_visits:
                    continue
                is_line_through_convex_hull_sf = False
                for target_compare, hull_sf in targets_border_sf.items():
                    line_points = points_along_line(target_points_of_view_sf[target][prior_idx, :3],
                                                    target_points_of_view_sf[target][max_idx, :3], 50)
                    is_line_through_convex_hull_sf = is_line_through_convex_hull(hull_sf, line_points)
                    if is_line_through_convex_hull_sf:
                        break
                if is_line_through_convex_hull_sf:
                    continue

                idx_list.append(max_idx)
                distance_p2p = np.linalg.norm(
                    target_points_of_view_sf[target][prior_idx, :3] - target_points_of_view_sf[target][max_idx, :3])
                total_distance = distance_p2p + total_distance
                CA += contribution
                visits_to_position[max_idx] += 1
                prior_idx_s = length + prior_idx
                max_idx_s = length + max_idx
                S[target][-1].append((subgroup_idx,
                                      prior_idx,
                                      max_idx,
                                      distance_p2p,
                                      total_distance,
                                      CA,
                                      prior_idx_s,
                                      max_idx_s))
                prior_idx = max_idx
            if iteration >= max_iter - 1:
                print('Decrease CA_max')
            if len(S[target][-1]) == 0:
                S[target].pop()
            else:
                subgroup_idx += 1
        length += points.shape[0]
        is_first_target = False
        cont_target += 1
    return S


# def subgroup_formation(targets_border_sf: dict, points_of_view_contribution_sf: dict,
#                        target_points_of_view_sf: dict, positions_sf: dict) -> dict:
#     print('Starting subgroup formation')
#     S = {}
#     contribution = 0
#     CA_max = 100  # Assuming CA_max and max_iter are defined somewhere
#     max_iter = 100
#     max_visits = 50
#     for target, points in target_points_of_view_sf.items():
#         plotter_sf = pv.Plotter()
#         point_cloud = pv.PolyData(positions_sf[target])
#         plotter_sf.add_mesh(point_cloud, point_size=10)
#         S[target] = []
#         visits_to_position = np.zeros(points.shape[0])
#         indexes_of_ini_points = list(range(points.shape[0]))
#         random_points = sample(indexes_of_ini_points, len(indexes_of_ini_points))
#         show_number_of_points = 0
#         for i in random_points:
#             print(f'Point {show_number_of_points} of {len(indexes_of_ini_points)}')
#             show_number_of_points += 1
#             CA = 0
#             total_distance = 0
#             S[target].append([])
#             prior_idx = i
#             idx_list = [i]
#             iteration = 0
#             while CA < CA_max and iteration < max_iter:
#                 iteration += 1
#                 indexes_of_points = np.random.randint(low=0, high=points.shape[0], size=10)
#                 max_contribution = 0
#                 for index in indexes_of_points:
#                     if index in idx_list:
#                         continue
#                     is_point_inside_sf = any(is_point_inside(points[index, :3],
#                                                              hull_sf) for hull_sf in targets_border_sf.values())
#                     all_points_line = points_along_line(target_points_of_view_sf[target][prior_idx, :3],
#                                                         target_points_of_view_sf[target][index, :3],
#                                                         50)
#                     test_bla = is_line_through_convex_hull(hull_sf, all_points_line)
#                     is_line_through_convex_hull_sf = (
#                         any(is_line_through_convex_hull(hull_sf, all_points_line)
#                             for hull_sf in targets_border_sf.values() if hull_sf != targets_border_sf[target]))
#                     points_line = pv.PolyData(all_points_line)
#                     plotter_sf.add_mesh(points_line, color='red')
#                     plotter_sf.show()
#                     if is_point_inside_sf or is_line_through_convex_hull_sf:
#                         continue
#                     distance_p2p = np.linalg.norm(target_points_of_view_sf[target][prior_idx, :3] -
#                                                   target_points_of_view_sf[target][index, :3])
#                     contribution = abs(points_of_view_contribution_sf[target][index] - distance_p2p)
#                     if contribution > max_contribution:
#                         max_idx = index
#                         max_contribution = contribution
#                 if max_contribution == 0:
#                     continue
#                 if visits_to_position[max_idx] > max_visits:
#                     continue
#                 idx_list.append(max_idx)
#                 distance_p2p = np.linalg.norm(
#                     target_points_of_view_sf[target][prior_idx, :3] - target_points_of_view_sf[target][max_idx, :3])
#                 total_distance += distance_p2p
#                 CA += max_contribution
#                 S[target][-1].append((prior_idx, max_idx, distance_p2p, total_distance, CA))
#                 visits_to_position[max_idx] += 1
#                 prior_idx = max_idx
#             if len(S[target][-1]) == 0:
#                 S[target].pop()
#     return S


def find_route(S_fr: dict, points_of_view_contribution_sf: dict = None, target_points_of_view_sf: dict = None):
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
    route_result_points_gpfr = np.empty(0)
    for point_pair in route_points_gpfr:
        if len(route_result_points_gpfr) == 0:
            route_result_points_gpfr = points_table_gpfr[point_pair[0]]
            continue
        route_result_points_gpfr = np.row_stack((route_result_points_gpfr, points_table_gpfr[point_pair[1]]))
    return route_result_points_gpfr


def save_points(route_sp: dict, targets_points_of_view_sr: dict):
    print('Starting saving ...')
    route_points = np.empty([0, 6])
    for target, data_s in route_sp.items():
        for data in data_s:
            point_start = targets_points_of_view_sr[target][data[0]]
            point_end = targets_points_of_view_sr[target][data[1]]
            route_points = np.row_stack((route_points, point_end))
    np.savetxt('positions.csv', route_points, delimiter=',')


def initializations(copp) -> tuple:
    """
    Function to get the points from CoppeliaSim. The points of each object can not be at the same plane, at least one
    must be a different plane. On CoppeliaSim you must add discs around the object to form a convex hull these points
    must call Disc[0], Disc[1], ... , Disc[n]. These points must be son of a plane named O[0], O[1], ... , O[n]. These
    objects in CoppeliaSim scene must have the property Object is model on window Scene Object Properties checked. To
    access these properties you can only double-click on object.
    :return:
    """
    positions = {}
    j = 0
    targets_hull_i = {}
    centroid_points_i = {}
    radius_i = {}
    for object_name_i in settings['object names']:
        # copp.handles[f'./O[{j}]'] = copp.sim.getObject(":/O", {'index': j, 'noError': True})
        copp.handles[object_name_i] = copp.sim.getObject(f"./{object_name_i}")
        if copp.handles[object_name_i] < 0:
            break
        positions[object_name_i] = np.empty([0, 3])
        i = 0
        while True:
            points_names = f"./{object_name_i}/Disc"
            handle = copp.sim.getObject(points_names, {'index': i, 'noError': True})
            if handle < 0:
                break
            positions[object_name_i] = np.row_stack((positions[object_name_i],
                                                     copp.sim.getObjectPosition(handle,
                                                                                copp.sim.handle_world)))
            i += 1

        targets_hull_i[object_name_i] = Delaunay(positions[object_name_i])
        centroid_points_i[object_name_i], radius_i[object_name_i] = _centroid_poly(positions[object_name_i])
        j = j + 1

    return positions, targets_hull_i, centroid_points_i, radius_i


def _centroid_poly(poly: np.ndarray):
    T = Delaunay(poly).simplices
    n = T.shape[0]
    W = np.zeros(n)
    C = 0

    for m in range(n):
        sp = poly[T[m, :], :]
        sp += np.random.normal(0, 1e-10, sp.shape)
        W[m] = ConvexHull(sp).volume
        C += W[m] * np.mean(sp, axis=0)

    tmp_center = C / np.sum(W)
    max_distance = 0
    for m in range(n):
        sp = poly[T[m, :], :2]
        for spl in sp:
            distance = np.linalg.norm(spl - tmp_center[:2])
            if distance > max_distance:
                max_distance = distance

    return tmp_center, max_distance


def get_geometric_objects_cell(geometric_objects):
    for i in range(geometric_objects.n_cells):
        yield geometric_objects.get_cell(i)


def find_normal_vector(point1, point2, point3):
    vec1 = np.array(point2) - np.array(point1)
    vec2 = np.array(point3) - np.array(point1)
    cross_vec = np.cross(vec1, vec2)
    return cross_vec / np.linalg.norm(cross_vec)


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

    return roll, pitch, yaw


def compute_radial_area():
    print('Starting computing radial area')


def draw_cylinders_hemispheres(centroid_points_pf: dict,
                               radius_pf: dict,
                               target_points_pf: dict) -> tuple[
    dict[Any, ndarray[Any, dtype[Any]] | ndarray[Any, dtype[floating[_64Bit] | float_]]], dict[
        Any, list[float] | list[Any]], list[ndarray[Any, dtype[Any]]]]:
    """
    Draw the hemispheres and the cylinders around the object
    :param centroid_points_pf: Dictionary of arrays of points with the central points of the objects
    :param radius_pf: Computed radius of the cylinders around each object
    :param target_points_pf: Computed points of the convex hull of the objects
    :return: vector_of_points: Dictionary with points around each object
    :return: Dictionary of weight of each point
    """
    print('Starting showing data')
    # Create a plotter
    plotter = pv.Plotter()
    vector_points_pf = {}  # Dictionary with points of view around each object
    vector_points_weight_pf = {}  # Dictionary of weights to each point
    central_area_computed = False  # Verify if the weight to each point in the normal was computed
    computed_area_by_hemisphere = []  # Stores the computed area to each point computed the first time
    is_included_first_group = False
    conversion_table = []
    for target in centroid_points_pf.keys():
        cy_direction = np.array([0, 0, 1])
        cy_hight = height_proportion * (np.max(target_points_pf[target][:, 2]) - np.min(target_points_pf[target][:, 2]))
        r_mesh = radius_pf[target]
        h = np.cos(np.pi / n_resolution) * r_mesh
        # l = np.sqrt(np.abs(4 * h ** 2 - 4 * r_mesh ** 2))

        # Find the radius of the spheres
        meshes = draw_cylinder_with_hemisphere(plotter,
                                               cy_direction,
                                               cy_hight,
                                               n_resolution,
                                               r_mesh,
                                               centroid_points_pf[target])
        cylinder = meshes['cylinder']['mesh']
        if not is_included_first_group:
            vector_points_pf[target] = np.array([-2.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            vector_points_weight_pf[target] = [0.0]
            conversion_table.append(np.array([-2.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
            is_included_first_group = True
        else:
            vector_points_pf[target] = np.empty([0, 6])
            vector_points_weight_pf[target] = []
        count_hemisphere = 0
        route_radius_dch = int(np.fix(max_route_radius // points_per_unit))
        weights = (route_radius_dch - 1) * [0.0]
        for cell in get_geometric_objects_cell(cylinder):
            # print(f'{count_hemisphere} of {cylinder.n_cells}')
            hemisphere_radius = meshes['hemispheres'][count_hemisphere]['radius']  #
            pos_cell = cell.center
            points_cell = cell.points[:3]
            norm_vec = find_normal_vector(*points_cell)
            roll, pitch, yaw = euler_angles_from_normal(-norm_vec)
            for k in range(1, route_radius_dch):
                camera_distance = ((points_per_sphere * k) + 1) * hemisphere_radius
                point_position = pos_cell + camera_distance * norm_vec
                # spherical_area_dc = 2 * np.pi * hemisphere_radius ** 2
                # if not central_area_computed:
                #     if not reach_maximum:
                if (count_hemisphere == 0 or count_hemisphere == n_resolution or
                        count_hemisphere == cylinder.n_cells - n_resolution):
                    spherical_area_dc, reach_maximum, frustum_planes, cam_pos = (
                        compute_central_hemisphere_area(norm_vec,
                                                        pos_cell,
                                                        hemisphere_radius,
                                                        camera_distance,
                                                        plotter))
                    area = get_side_hemisphere_area(cylinder.n_cells,
                                                    meshes,
                                                    frustum_planes,
                                                    count_hemisphere)
                    weight = spherical_area_dc + area
                    weights[k - 1] = weight
                else:
                    weight = weights[k - 1]
                vector_points_pf[target] = np.row_stack((vector_points_pf[target],
                                                         np.concatenate((point_position,
                                                                         np.array([yaw, pitch, roll])))))
                conversion_table.append(np.concatenate((point_position, np.array([yaw, pitch, roll]))))
                vector_points_weight_pf[target].append(weight)
            count_hemisphere += 1

        points0 = vector_points_pf[target][:, :3]
        point_cloud0 = pv.PolyData(points0)
        plotter.add_mesh(point_cloud0)

        # cylinder.plot(show_edges=True)
        plotter.add_mesh(cylinder, show_edges=True)

        points = target_points_pf[target]
        point_cloud = pv.PolyData(points)
        plotter.add_mesh(point_cloud)

    # plotter.show()
    return vector_points_pf, vector_points_weight_pf, conversion_table


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


def get_side_hemisphere_area(count_plane_gsha: int,
                             meshes_gsha: dict,
                             frustum_planes: list,
                             central_hemisphere_gsha: int) -> float:
    # print('Starting getting side hemisphere area')

    # tmpidxs = [(central_hemisphere_gsha + count_idx) % n_resolution + (central_hemisphere_gsha // n_resolution) * n_resolution for count_idx in
    #            range(1, 4)] * 2
    # if central_hemisphere_gsha > n_resolution:
    #     tmpidxs.extend([idx - n_resolution for idx in tmpidxs])
    # if central_hemisphere_gsha < count_plane_gsha - n_resolution:
    #     tmpidxs.extend([idx + n_resolution for idx in tmpidxs])
    #
    # area = 0
    # for hemisphere_idx in tmpidxs:
    #     ct_pt = np.array(meshes_gsha['hemispheres'][hemisphere_idx]['center'])
    #     is_in = False
    #     for plane_gsha in frustum_planes:
    #         distance = (abs(np.dot(plane_gsha[:3], ct_pt) + plane_gsha[3]) /
    #                     np.sqrt(plane_gsha[0] ** 2 + plane_gsha[1] ** 2 + plane_gsha[2] ** 2))
    #         if distance < meshes_gsha['hemispheres'][hemisphere_idx]['radius']:
    #             is_in = True
    #             x = (-plane_gsha[3] - ct_pt[1] * plane_gsha[1] - ct_pt[2] * plane_gsha[2]) / plane_gsha[0]
    #             point_pi = np.array([x, ct_pt[1], ct_pt[2]])
    #             intersection_points = intersect_plane_sphere(np.array(plane_gsha[:3]), point_pi,
    #                                                          np.array(
    #                                                              meshes_gsha['hemispheres'][hemisphere_idx]['center']),
    #                                                          meshes_gsha['hemispheres'][hemisphere_idx]['radius'])
    #             break
    #
    #     alpha = 1 / (
    #                 1 + np.linalg.norm(ct_pt - np.array(meshes_gsha['hemispheres'][central_hemisphere_gsha]['center'])))
    #
    #     if not is_in:
    #         area += 2 * alpha * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'] ** 2
    #     else:
    #         if point_between_planes(ct_pt, np.array(frustum_planes)):
    #             area += alpha * 2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'] * np.linalg.norm(
    #                 intersection_points[0] - intersection_points[1])
    #         else:
    #             area += alpha * (2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'] *
    #                              np.linalg.norm(intersection_points[0] - intersection_points[1]) +
    #                              2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'])
    #
    # return area

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
        # hemisphere_idx = (central_hemisphere_gsha - count_neighbor) % 24 + (central_hemisphere_gsha // 24) * 24
        # distance = (abs(np.dot(plane_eq_gsha[:3], meshes_gsha['hemispheres'][hemisphere_idx]['center']) + plane_eq_gsha[3]) /
        #             np.sqrt(plane_eq_gsha[0] ** 2 +
        #                     plane_eq_gsha[1] ** 2 +
        #                     plane_eq_gsha[2] ** 2))
        # plane_eq_gsha = np.array(frustum_planes[2])
        # hemisphere_idx = (central_hemisphere_gsha + count_neighbor) % 24 + (central_hemisphere_gsha // 24) * 24
        # x = (-plane_eq_gsha[3] - meshes_gsha['hemispheres'][hemisphere_idx]['center'][1] * plane_eq_gsha[1] -
        #      meshes_gsha['hemispheres'][hemisphere_idx]['center'][2] * plane_eq_gsha[2]) / plane_eq_gsha[0]
        # point_pi1 = np.array([x, meshes_gsha['hemispheres'][hemisphere_idx]['center'][1],
        #                       meshes_gsha['hemispheres'][hemisphere_idx]['center'][2]])
        # plotter1 = pv.Plotter()
        # plotter1.add_mesh(pv.Plane(center=point_pi, direction=plane_eq_gsha[:3], i_size=4, j_size=4))
        # plotter1.add_mesh(pv.PolyData(point_pi), color='green', point_size=10)
        # ct_pt = np.array(meshes_gsha['hemispheres'][hemisphere_idx]['center'])
        # plotter1.add_mesh(pv.PolyData(ct_pt), color='red', point_size=10)
        # plotter1.add_mesh(pv.Sphere(radius=meshes_gsha['hemispheres'][hemisphere_idx]['radius'], center=ct_pt))
        # plotter1.show()
        # P1 = ct_pt - point_pi1
        # Projection_P1inP21 = (P1 @ point_pi1)/(np.linalg.norm(point_pi1)**2)*point_pi1
        # dir1 = P1 - Projection_P1inP21

        # plane_eq_gsha2 = np.array(frustum_planes[3])
        # x = (-plane_eq_gsha[3] - meshes_gsha['hemispheres'][hemisphere_idx]['center'][1] * plane_eq_gsha[1] -
        #      meshes_gsha['hemispheres'][hemisphere_idx]['center'][2] * plane_eq_gsha[2]) / plane_eq_gsha[0]
        # point_pi2 = np.array([x, meshes_gsha['hemispheres'][hemisphere_idx]['center'][1],
        #                      meshes_gsha['hemispheres'][hemisphere_idx]['center'][2]])
        # P2 = ct_pt - point_pi2
        # Projection_P1inP22 = (P2 @ point_pi2)/(np.linalg.norm(point_pi2)**2)*point_pi2
        # dir2 = P2 - Projection_P1inP22
        # hemisphere_idx = 1
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
        alpha = 1  # / (
        # 1 + np.linalg.norm(ct_pt - np.array(meshes_gsha['hemispheres'][central_hemisphere_gsha]['center'])))
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

    # distance2 = (abs(np.dot(plane_eq_gsha2[:3], meshes_gsha['hemispheres'][hemisphere_idx]['center']) + plane_eq_gsha2[
    #     3]) /
    #             np.sqrt(plane_eq_gsha2[0] ** 2 +
    #                     plane_eq_gsha2[1] ** 2 +
    #                     plane_eq_gsha2[2] ** 2))

    # distance = np.dot(plane_eq_gsha[:3], ct_pt - point_pi1)

    # if distance1 < meshes_gsha['hemispheres'][hemisphere_idx]['radius'] or distance2 < meshes_gsha['hemispheres'][hemisphere_idx]['radius']:
    #     print(f'The plane {count_plane_gsha} has intersection with hemisphere 1')
    #     if distance1 < distance2:
    #         x = (-plane_eq_gsha[3] - meshes_gsha['hemispheres'][hemisphere_idx]['center'][1] * plane_eq_gsha[1] -
    #              meshes_gsha['hemispheres'][hemisphere_idx]['center'][2] * plane_eq_gsha[2]) / plane_eq_gsha[0]
    #         point_pi = np.array([x, meshes_gsha['hemispheres'][hemisphere_idx]['center'][1],
    #                              meshes_gsha['hemispheres'][hemisphere_idx]['center'][2]])
    #         intersection_points = intersect_plane_sphere(np.array(plane_eq_gsha[:3]),
    #                                                      point_pi,
    #                                                      np.array(meshes_gsha['hemispheres'][hemisphere_idx]['center']),
    #                                                      meshes_gsha['hemispheres'][hemisphere_idx]['radius'])
    #     else:
    #         x = (-plane_eq_gsha2[3] - meshes_gsha['hemispheres'][hemisphere_idx]['center'][1] * plane_eq_gsha2[1] -
    #              meshes_gsha['hemispheres'][hemisphere_idx]['center'][2] * plane_eq_gsha2[2]) / plane_eq_gsha2[0]
    #         point_pi = np.array([x, meshes_gsha['hemispheres'][hemisphere_idx]['center'][1],
    #                              meshes_gsha['hemispheres'][hemisphere_idx]['center'][2]])
    #         intersection_points = intersect_plane_sphere(np.array(plane_eq_gsha2[:3]),
    #                                                      point_pi,
    #                                                      np.array(meshes_gsha['hemispheres'][hemisphere_idx]['center']),
    #                                                      meshes_gsha['hemispheres'][hemisphere_idx]['radius'])
    #
    #     if point_between_planes(ct_pt, plane_eq_gsha, plane_eq_gsha2):
    #         area += 2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'] * np.linalg.norm(
    #             intersection_points[0] - intersection_points[1])
    #     else:
    #         area += (2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'] *
    #                 np.linalg.norm(intersection_points[0] - intersection_points[1]) +
    #                 2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'])
    # else:
    #     # x = (-plane_eq_gsha[3] - meshes_gsha['hemispheres'][hemisphere_idx]['center'][1] * plane_eq_gsha[1] -
    #     #      meshes_gsha['hemispheres'][hemisphere_idx]['center'][2] * plane_eq_gsha[2]) / plane_eq_gsha[0]
    #     # point_pi = np.array([x, meshes_gsha['hemispheres'][hemisphere_idx]['center'][1],
    #     #                      meshes_gsha['hemispheres'][hemisphere_idx]['center'][2]])
    #     if not point_between_planes(ct_pt, plane_eq_gsha, plane_eq_gsha2):
    #         area += 2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'] ** 2
    #     else:
    #         area += 0
    #
    # hemisphere_idx = 23  # works for 24, 23, 22
    # distance = (abs(np.dot(plane_eq_gsha[:3], meshes_gsha['hemispheres'][hemisphere_idx]['center']) + plane_eq_gsha[3]) /
    #             np.sqrt(plane_eq_gsha[0] ** 2 +
    #                     plane_eq_gsha[1] ** 2 +
    #                     plane_eq_gsha[2] ** 2))
    # plane_eq_gsha = frustum_planes[2]
    # hemisphere_idx = (central_hemisphere_gsha - count_neighbor) % 24 + (central_hemisphere_gsha // 24) * 24
    # x = (-plane_eq_gsha[3] - plane_eq_gsha[1] - plane_eq_gsha[2]) / plane_eq_gsha[0]
    # point_pi1 = np.array([x, 1.0, 1.0])
    # plotter1 = pv.Plotter()
    # plotter1.add_mesh(pv.Plane(center=point_pi, direction=plane_eq_gsha[:3], i_size=4, j_size=4))
    # plotter1.add_mesh(pv.PolyData(point_pi), color='green', point_size=10)
    # ct_pt = np.array(meshes_gsha['hemispheres'][hemisphere_idx]['center'])
    # plotter1.add_mesh(pv.PolyData(ct_pt), color='red', point_size=10)
    # plotter1.add_mesh(pv.Sphere(radius=meshes_gsha['hemispheres'][hemisphere_idx]['radius'], center=ct_pt))
    # plotter1.show()
    # P1 = ct_pt - point_pi1
    # Projection_P1inP21 = (P1 @ point_pi1)/(np.linalg.norm(point_pi1)**2)*point_pi1
    # dir1 = P1 - Projection_P1inP21
    # plane_eq_gsha = np.array(frustum_planes[3])
    # x = (-plane_eq_gsha[3] - plane_eq_gsha[1] - plane_eq_gsha[2]) / plane_eq_gsha[0]
    # point_pi2 = np.array([x, 1.0, 1.0])
    # P2 = ct_pt - point_pi2
    # Projection_P1inP22 = (P2 @ point_pi2)/(np.linalg.norm(point_pi2)**2)*point_pi2
    # dir2 = P2 - Projection_P1inP22
    #
    # distance = (abs(np.dot(plane_eq_gsha[:3], meshes_gsha['hemispheres'][hemisphere_idx]['center']) + plane_eq_gsha[
    #     3]) /
    #             np.sqrt(plane_eq_gsha[0] ** 2 +
    #                     plane_eq_gsha[1] ** 2 +
    #                     plane_eq_gsha[2] ** 2))
    # if distance < meshes_gsha['hemispheres'][hemisphere_idx]['radius']:
    #     print(f'The plane {count_plane_gsha} has intersection with hemisphere 1')
    #     x = (-plane_eq_gsha[3] - meshes_gsha['hemispheres'][hemisphere_idx]['center'][1] * plane_eq_gsha[1] -
    #          meshes_gsha['hemispheres'][hemisphere_idx]['center'][2] * plane_eq_gsha[2]) / plane_eq_gsha[0]
    #     point_pi = np.array([x, meshes_gsha['hemispheres'][hemisphere_idx]['center'][1],
    #                          meshes_gsha['hemispheres'][hemisphere_idx]['center'][2]])
    #     intersection_points = intersect_plane_sphere(np.array(plane_eq_gsha[:3]),
    #                                                  point_pi,
    #                                                  np.array(meshes_gsha['hemispheres'][hemisphere_idx]['center']),
    #                                                  meshes_gsha['hemispheres'][hemisphere_idx]['radius'])
    #     if point_pi[0] < meshes_gsha['hemispheres'][hemisphere_idx]['center'][0]:
    #         area += 2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'] * np.linalg.norm(
    #             intersection_points[0] - intersection_points[1])
    #     else:
    #         area += (2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'] *
    #                 np.linalg.norm(intersection_points[0] - intersection_points[1]) +
    #                 2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'])
    # else:
    #     x = (-plane_eq_gsha[3] - meshes_gsha['hemispheres'][hemisphere_idx]['center'][1] * plane_eq_gsha[1] -
    #          meshes_gsha['hemispheres'][hemisphere_idx]['center'][2] * plane_eq_gsha[2]) / plane_eq_gsha[0]
    #     point_pi = np.array([x, meshes_gsha['hemispheres'][hemisphere_idx]['center'][1],
    #                          meshes_gsha['hemispheres'][hemisphere_idx]['center'][2]])
    #     if point_pi[0] > meshes_gsha['hemispheres'][hemisphere_idx]['center'][0]:
    #         area += 2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'] ** 2
    #     else:
    #         area += 0
    #
    # if count_neighbor * 24 - 24 > 0:  # and hemisphere_idx == central_hemisphere_gsha + 25:
    #     # hemisphere_idx = 25  # Works for 25, 49
    #     plane_eq_gsha = frustum_planes[1]
    #     hemisphere_idx = central_hemisphere_gsha - count_neighbor * 24
    #     distance = (abs(np.dot(plane_eq_gsha[:3], meshes_gsha['hemispheres'][hemisphere_idx]['center']) +
    #                     plane_eq_gsha[
    #                         3]) /
    #                 np.sqrt(plane_eq_gsha[0] ** 2 +
    #                         plane_eq_gsha[1] ** 2 +
    #                         plane_eq_gsha[2] ** 2))
    #     if distance < meshes_gsha['hemispheres'][hemisphere_idx]['radius']:
    #         print(f'The plane {count_plane_gsha} has intersection with hemisphere 1')
    #         y = (-plane_eq_gsha[3] - meshes_gsha['hemispheres'][hemisphere_idx]['center'][0] * plane_eq_gsha[0] -
    #              meshes_gsha['hemispheres'][hemisphere_idx]['center'][2] * plane_eq_gsha[2]) / plane_eq_gsha[1]
    #         point_pi = np.array([meshes_gsha['hemispheres'][hemisphere_idx]['center'][0], y,
    #                              meshes_gsha['hemispheres'][hemisphere_idx]['center'][2]])
    #         intersection_points = intersect_plane_sphere(np.array(plane_eq_gsha[:3]),
    #                                                      point_pi,
    #                                                      np.array(
    #                                                          meshes_gsha['hemispheres'][hemisphere_idx]['center']),
    #                                                      meshes_gsha['hemispheres'][hemisphere_idx]['radius'])
    #         if point_pi[1] > meshes_gsha['hemispheres'][hemisphere_idx]['center'][1]:
    #             area += 2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'] * np.linalg.norm(
    #                 intersection_points[0] - intersection_points[1])
    #         else:
    #             area += (2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'] *
    #                     np.linalg.norm(intersection_points[0] - intersection_points[1]) +
    #                     2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'])
    #     else:
    #         y = (-plane_eq_gsha[3] - meshes_gsha['hemispheres'][hemisphere_idx]['center'][0] * plane_eq_gsha[0] -
    #              meshes_gsha['hemispheres'][hemisphere_idx]['center'][2] * plane_eq_gsha[2]) / plane_eq_gsha[1]
    #         point_pi = np.array([meshes_gsha['hemispheres'][hemisphere_idx]['center'][0], y,
    #                              meshes_gsha['hemispheres'][hemisphere_idx]['center'][2]])
    #         if point_pi[1] < meshes_gsha['hemispheres'][hemisphere_idx]['center'][1]:
    #             area += 2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'] ** 2
    #         else:
    #             area += 0
    #
    # hemisphere_idx = 25  # Works for 25, 49
    # plane_eq_gsha = frustum_planes[0]
    # hemisphere_idx = central_hemisphere_gsha + count_neighbor * 24
    # distance = (abs(np.dot(plane_eq_gsha[:3], meshes_gsha['hemispheres'][hemisphere_idx]['center']) + plane_eq_gsha[
    #     3]) /
    #             np.sqrt(plane_eq_gsha[0] ** 2 +
    #                     plane_eq_gsha[1] ** 2 +
    #                     plane_eq_gsha[2] ** 2))
    # if distance < meshes_gsha['hemispheres'][hemisphere_idx]['radius']:
    #     print(f'The plane {count_plane_gsha} has intersection with hemisphere 1')
    #     y = (-plane_eq_gsha[3] - meshes_gsha['hemispheres'][hemisphere_idx]['center'][0] * plane_eq_gsha[0] -
    #          meshes_gsha['hemispheres'][hemisphere_idx]['center'][2] * plane_eq_gsha[2]) / plane_eq_gsha[1]
    #     point_pi = np.array([meshes_gsha['hemispheres'][hemisphere_idx]['center'][0], y,
    #                          meshes_gsha['hemispheres'][hemisphere_idx]['center'][2]])
    #     intersection_points = intersect_plane_sphere(np.array(plane_eq_gsha[:3]),
    #                                                  point_pi,
    #                                                  np.array(meshes_gsha['hemispheres'][hemisphere_idx]['center']),
    #                                                  meshes_gsha['hemispheres'][hemisphere_idx]['radius'])
    #     if point_pi[1] < meshes_gsha['hemispheres'][hemisphere_idx]['center'][1]:
    #         area += 2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'] * np.linalg.norm(
    #             intersection_points[0] - intersection_points[1])
    #     else:
    #         area += (2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'] *
    #                 np.linalg.norm(intersection_points[0] - intersection_points[1]) +
    #                 2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'])
    # else:
    #     y = (-plane_eq_gsha[3] - meshes_gsha['hemispheres'][hemisphere_idx]['center'][0] * plane_eq_gsha[0] -
    #          meshes_gsha['hemispheres'][hemisphere_idx]['center'][2] * plane_eq_gsha[2]) / plane_eq_gsha[1]
    #     point_pi = np.array([meshes_gsha['hemispheres'][hemisphere_idx]['center'][0], y,
    #                          meshes_gsha['hemispheres'][hemisphere_idx]['center'][2]])
    #     if point_pi[1] > meshes_gsha['hemispheres'][hemisphere_idx]['center'][1]:
    #         area += 2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'] ** 2
    #     else:
    #         area += 0

    # return area


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


def quadcopter_control_direct_points(sim, client, vision_handle: int,
                                     route_qc: ndarray, filename_qcdp: str, directory_name_qcdp: str):
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

    # filename_qcdp = (settings['path'] + 'scene1/' + settings['filename'] + '_' + day + '_' + month + '_' + year + '_')
    for point_qcdp in route_qc:
        pos = list(point_qcdp[:3])
        # next_point_handle = sim.getObject('./new_target')
        # sim.setObjectPosition(next_point_handle, pos)
        # camera_orientation = sim.yawPitchRollToAlphaBetaGamma(point_qcdp[3], point_qcdp[4], point_qcdp[5])
        orientation = list(np.deg2rad(point_qcdp[3:]))
        orientation_angles = [0.0, 0.0, orientation[0]]
        # sim.setObjectPosition(vision_handle, pos)
        # sim.setObjectOrientation(vision_handle, camera_orientation)
        quadcopter_handle = sim.getObject('./base_vision')
        sim.setObjectPosition(quadcopter_handle, pos, sim.handle_world)
        sim.setObjectOrientation(quadcopter_handle, orientation_angles, sim.handle_world)
        # sim.setObjectOrientation(quad_target_handle, [0.0, 0.0, orientation[0]], sim.handle_world)
        # orientation_angles = sim.yawPitchRollToAlphaBetaGamma(orientation[0], orientation[2], orientation[1])
        # pos = sim.getObjectPosition(quad_base_handle, sim.handle_world)
        # camera_handle = sim.getObject('./O[0]/Cone[19]')
        # sim.setObjectOrientation(camera_handle, orientation_angles)
        # sim.setObjectPosition(camera_handle, pos)
        # client.step()
        # sim.setObjectOrientation(quad_target_handle, sim.handle_world, orientation_angles)

        total_time = sim.getSimulationTime() + settings['total simulation time']
        # stabilized = False
        while sim.getSimulationTime() < total_time:
            # diff_pos = np.subtract(pos, sim.getObjectPosition(quad_base_handle, sim.handle_world))
            # norm_diff_pos = np.linalg.norm(diff_pos)
            # if norm_diff_pos > 0.5:
            #     delta_pos = 0.1 * diff_pos
            #     new_pos = list(sim.getObjectPosition(quad_base_handle, sim.handle_world) + delta_pos)
            # else:
            #     new_pos = pos
            #
            # sim.setObjectPosition(quad_target_handle, new_pos)
            # diff_ori = np.subtract(orientation_angles,
            #                        sim.getObjectOrientation(quad_base_handle, sim.handle_world))
            # norm_diff_ori = np.linalg.norm(diff_ori)
            #
            # if norm_diff_ori > 0.08:
            #     delta_ori = 0.3 * diff_ori
            #     new_ori = list(sim.getObjectOrientation(quad_base_handle, sim.handle_world) + delta_ori)
            # else:
            #     new_ori = orientation_angles
            # sim.setObjectOrientation(quad_target_handle, new_ori)
            # t_stab = sim.getSimulationTime() + settings['time to stabilize']
            # while sim.getSimulationTime() < t_stab:
            #     diff_pos = np.subtract(new_pos,
            #                            sim.getObjectPosition(quad_base_handle, sim.handle_world))
            #     diff_ori = np.subtract(new_ori,
            #                            sim.getObjectOrientation(quad_base_handle, sim.handle_world))
            #     norm_diff_pos = np.linalg.norm(diff_pos)
            #     norm_diff_ori = np.linalg.norm(diff_ori)
            #     if norm_diff_pos < 0.1 and norm_diff_ori < 0.05:
            #         stabilized = True
            #         break
            #     client.step()
            # diff_pos = np.subtract(pos, sim.getObjectPosition(quad_base_handle, sim.handle_world))
            # diff_ori = np.subtract(orientation_angles, sim.getObjectOrientation(quad_base_handle, sim.handle_world))
            # norm_diff_pos = np.linalg.norm(diff_pos)
            # norm_diff_ori = np.linalg.norm(diff_ori)
            # if norm_diff_pos < 0.1 and norm_diff_ori < 0.05:
            #     stabilized = True
            #     break
            client.step()

        get_image(sim, count_image, filename_qcdp, vision_handle, directory_name_qcdp)
        count_image += 1
        # if not stabilized:
        #     print('Time short')


def compute_edge_weight_matrix(S_cewm: dict, targets_points_of_view_cewm) -> ndarray:
    print('Starting computing distance matrix')
    edge_weight_matrix_cewm = np.empty(0)
    i = 0
    j = 0
    # count_target = 0
    for target_cewm, S_cewm_start in S_cewm.items():
        if i == 0 and j == 0:
            edge_weight_matrix_cewm = np.zeros([2 * ((len(S_cewm_start) - 1) * len(settings['object names'])) - 1,
                                                2 * ((len(S_cewm_start) - 1) * len(settings['object names'])) - 1])
        for Si_cewm_start in S_cewm_start:
            j = 0
            # count_target_i = 0
            for target_cewm_i, S_cewm_end in S_cewm.items():
                for Si_cewm_end in S_cewm_end:
                    idx1 = Si_cewm_start[-1][1]  # - count_target*targets_points_of_view_cewm[target_cewm].shape[0]
                    idx2 = Si_cewm_end[0][1]  # - count_target_i*targets_points_of_view_cewm[target_cewm_i].shape[0]
                    pt1 = targets_points_of_view_cewm[target_cewm][idx1]
                    pt2 = targets_points_of_view_cewm[target_cewm_i][idx2]
                    if i != j:
                        edge_weight_matrix_cewm[i, j] = np.linalg.norm(pt1 - pt2)
                    j += 1
                # count_target_i += 1
            i += 1
        # count_target += 1
    i -= 1
    j -= 1
    edge_weight_matrix_cewm = edge_weight_matrix_cewm[:i, :j]
    return edge_weight_matrix_cewm


def ConvertArray2String(fileCA2S, array: ndarray):
    np.set_printoptions(threshold=10000000000)
    array_str = np.array2string(array, precision=5)
    array_str = array_str.replace('\n', '')
    array_str = array_str.replace('[', '')
    array_str = array_str.replace(']', '\n')
    array_str = array_str[:-1]
    fileCA2S.write(array_str)
    return fileCA2S


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


fieldnames = ['NAME: ', 'TYPE: ', 'COMMENT: ', 'DIMENSION: ', 'TMAX: ', 'START_CLUSTER: ', 'END_CLUSTER: ',
              'CLUSTERS: ', 'SUBGROUPS: ', 'DUBINS_RADIUS: ', 'EDGE_WEIGHT_TYPE: ', 'EDGE_WEIGHT_FORMAT: ',
              'EDGE_WEIGHT_SECTION', 'GTSP_SUBGROUP_SECTION: ', 'GTSP_CLUSTER_SECTION: ']


def copy_file(source_path, destination_path):
    try:
        # Copy the file from source_path to destination_path
        shutil.copy(source_path, destination_path)
        print(f"File copied successfully from {source_path} to {destination_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def write_problem_file(dir_wpf: str, filename_wpf: str, edge_weight_matrix_wpf: ndarray, number_of_targets: int,
                       S_wpf: dict):
    print('Starting writing problem file')
    subgroup_count = -1
    complete_file_name = dir_wpf + filename_wpf + '.cops'
    # print(f'{complete_file_name=}')
    with open(complete_file_name, 'w') as copsfile:
        for field_wpf in fieldnames:
            if field_wpf == 'NAME: ':
                copsfile.write(field_wpf + filename_wpf + settings['directory name'] + '\n')
            if field_wpf == 'TYPE: ':
                copsfile.write(field_wpf + 'TSP\n')
            if field_wpf == 'COMMENT: ':
                copsfile.write(field_wpf + 'Optimization for reconstruction\n')
            if field_wpf == 'DIMENSION: ':
                copsfile.write(field_wpf + str(subgroup_count) + '\n')
            if field_wpf == 'TMAX: ':
                copsfile.write(field_wpf + str(T_max) + '\n')
            if field_wpf == 'START_CLUSTER: ':
                copsfile.write(field_wpf + '0\n')
            if field_wpf == 'END_CLUSTER: ':
                copsfile.write(field_wpf + '0\n')
            if field_wpf == 'CLUSTERS: ':
                copsfile.write(field_wpf + str(number_of_targets) + '\n')
            if field_wpf == 'SUBGROUPS: ':
                copsfile.write(field_wpf + str(subgroup_count) + '\n')
            if field_wpf == 'DUBINS_RADIUS: ':
                copsfile.write(field_wpf + '50' + '\n')
            if field_wpf == 'EDGE_WEIGHT_TYPE: ':
                copsfile.write(field_wpf + 'EXPLICIT' + '\n')
            if field_wpf == 'EDGE_WEIGHT_FORMAT: ':
                copsfile.write(field_wpf + 'FULL_MATRIX' + '\n')
            if field_wpf == 'EDGE_WEIGHT_SECTION':
                copsfile.write(field_wpf + '\n')
                ConvertArray2String(copsfile, edge_weight_matrix_wpf)
            if subgroup_count == -1:
                for target_wpf, S_spf in S_wpf.items():
                    for lS_spf in S_spf:
                        subgroup_count += 1

            if field_wpf == 'GTSP_SUBGROUP_SECTION: ':
                copsfile.write(field_wpf + 'cluster_id cluster_profit id-vertex-list' + '\n')
                # count_target = 0
                for target_wpf, S_spf in S_wpf.items():
                    for lS_spf in S_spf:
                        copsfile.write(f'{lS_spf[0][0]} {lS_spf[-1][-1]} {lS_spf[0][6]} ')
                        for vertex in lS_spf:
                            copsfile.write(f'{vertex[7]} ')
                        copsfile.write('\n')

            if field_wpf == 'GTSP_CLUSTER_SECTION: ':
                copsfile.write(field_wpf + 'set_id id-cluster-list\n')
                count_cluster = 0
                for target_wpf, S_spf in S_wpf.items():
                    copsfile.write(str(count_cluster) + ' ')
                    count_cluster += 1
                    for lS_spf in S_spf:
                        copsfile.write(str(lS_spf[0][0]) + ' ')
                        subgroup_count -= 1
                        if subgroup_count <= 0:
                            break
                    copsfile.write('\n')
                if subgroup_count == 0:
                    break

    copsfile.close()
    # copy_file('./' + filename_wpf + '.cops', 'C:/Users/dnune/OneDrive/Documentos/VerLab/COPS/datasets/' + filename_wpf + '.cops')


def execute_script(script_path):
    try:
        # Execute the script using subprocess
        process = subprocess.Popen(['python', 'C:/Users/dnune/OneDrive/Documentos/VerLab/COPS/tabu_search.py',
                                    '--path=./datasets/3dreconstructionPathPlanner.cops'], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

        # Wait for the process to finish
        stdout, stderr = process.communicate()

        # Check if there were any errors
        if process.returncode != 0:
            print("Error executing script:")
            print(stderr.decode('utf-8'))
        else:
            print("Script executed successfully.")
    except Exception as e:
        print("An error occurred:", e)


def read_route_csv_file(file_path, S_rrcf: dict, targets_points_of_vew_rrcf: dict) -> tuple[
    ndarray, float, list[ndarray]]:
    route_rrcf = np.empty([0, 6])
    route_by_group = [np.empty([0, 6])] * len(settings['object names'])
    travelled_distance = 0
    try:
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=';')
            for row in csv_reader:
                route_str = row[8]
    except Exception as e:
        print(f"An error occurred: {e}")
    chose_subgroups = ast.literal_eval(route_str.replace('  ', ','))
    bigger_idx = 0
    table_rrcf = []
    for target_rrcf, points_rrcf in targets_points_of_vew_rrcf.items():
        table_rrcf.append([target_rrcf, bigger_idx, bigger_idx + points_rrcf.shape[0]])
        bigger_idx += points_rrcf.shape[0] + 1
    count_group = 0
    is_group_zero = True
    is_group_zero_zero = True
    for S_idx_rrcf in chose_subgroups:
        for information_rrcf in table_rrcf:
            if information_rrcf[1] <= S_idx_rrcf <= information_rrcf[2]:
                is_first_element = True
                for group_rrcf in S_rrcf[information_rrcf[0]]:
                    for element in group_rrcf:
                        if element[0] == S_idx_rrcf:
                            pt_idx_prior = element[1]
                            pt_idx_post = element[2]
                            pt_prior_coordinates = targets_points_of_vew_rrcf[information_rrcf[0]][pt_idx_prior]
                            pt_post_coordinates = targets_points_of_vew_rrcf[information_rrcf[0]][pt_idx_post]
                            travelled_distance += np.linalg.norm(pt_post_coordinates[:3] - pt_prior_coordinates[:3])
                            if is_first_element:
                                route_rrcf = np.row_stack((route_rrcf, pt_prior_coordinates, pt_post_coordinates))
                                is_first_element = False
                                if is_group_zero_zero:
                                    is_group_zero_zero = False
                                else:
                                    route_by_group[count_group] = np.row_stack((route_by_group[count_group],
                                                                                pt_prior_coordinates,
                                                                                pt_post_coordinates))
                            else:
                                route_rrcf = np.row_stack((route_rrcf, pt_post_coordinates))
                                route_by_group[count_group] = np.row_stack(
                                    (route_by_group[count_group], pt_post_coordinates))
        if is_group_zero:
            is_group_zero = False
        else:
            count_group += 1
    print(f'{travelled_distance=}')
    route_rrcf = np.row_stack((route_rrcf, route_rrcf[0]))
    return route_rrcf, travelled_distance, route_by_group


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

    # Define the directory name
    directory_name = directory_name_gi

    # Specify the path where you want to create the directory
    path = settings['path']  # You can specify any desired path here

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

    filename = (full_path + '/' + file_name + '_' + day + '_' + month + '_' + hour + '_' + minute + '_' +
                str(sequence) + '.' + settings['extension'])

    # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
    # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
    # and color format is RGB triplets, whereas OpenCV uses BGR:
    img = cv.flip(cv.cvtColor(img, cv.COLOR_BGR2RGB), 0)

    cv.imwrite(filename, img)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # targets_border, targets_center, targets_points_of_view = create_sample_data()
    # points_of_view_contribution = camera_view_evaluation(targets_points_of_view)
    # S = subgroup_formation(targets_border, points_of_view_contribution, targets_points_of_view)
    # main_route = find_route(S)
    # save_points(main_route, targets_points_of_view)
    # points_of_view_contribution = camera_view_evaluation(targets_points_of_view)
    settings = parse_settings_file('config.yaml')
    CA_max = float(settings['CA_max'])
    max_route_radius = float(settings['max route radius'])
    points_per_sphere = int(settings['points per sphere'])
    height_proportion = float(settings['height proportion'])
    max_visits = int(settings['max visits'])
    max_iter = int(settings['max iter'])
    T_max = float(settings['T_max'])
    n_resolution = int(settings['n resolution'])
    points_per_unit = int(settings['points per unit'])

    copp = CoppeliaInterface()
    for experiment in range(10):
        positions, target_hull, centroid_points, radius = initializations(copp)
        targets_points_of_view, points_of_view_contribution, conversion_table = draw_cylinders_hemispheres(
            centroid_points,
            radius,
            positions)
        S = subgroup_formation(target_hull, points_of_view_contribution, targets_points_of_view, positions)
        edge_weight_matrix = compute_edge_weight_matrix(S, targets_points_of_view)
        write_problem_file('C:/Users/dnune/OneDrive/Documentos/VerLab/RoutePlanner/datasets/',
                           '3dreconstructionPathPlanner',
                           edge_weight_matrix,
                           3,
                           S)
        execute_script('script_path')
        main_route, travelled_distance_main, route_by_group = read_route_csv_file(
            'C:/Users/dnune/OneDrive/Documentos/VerLab/RoutePlanner/datasets/results/' +
            '3dreconstructionPathPlanner.csv', S, targets_points_of_view)

        # main_route = get_points_to_route(route, conversion_table)
        # main_route = find_route(S)
        # route_points = get_points_route(targets_points_of_view, main_route)
        # plot_route(centroid_points, radius, positions, route_points)
        #
        # copp.handles[settings['quadcopter name']] = copp.sim.getObject(settings['quadcopter name'])
        # copp.handles[settings['quadcopter base']] = copp.sim.getObject(settings['quadcopter base'])
        copp.handles[settings['vision sensor names']] = copp.sim.getObject(settings['vision sensor names'])
        filename = settings['filename']
        directory_name = settings['directory name'] + f'_exp_{experiment}'
        quadcopter_control_direct_points(copp.sim,
                                         copp.client,
                                         copp.handles[settings['vision sensor names']],
                                         main_route,
                                         filename,
                                         directory_name)
        # Get the current date and time
        current_datetime = datetime.datetime.now()
        colmap_folder = settings['colmap folder']
        month = str(current_datetime.month)
        day = str(current_datetime.day)
        hour = str(current_datetime.hour)
        minute = str(current_datetime.minute)
        workspace_folder = os.path.join(settings['workspace folder'], f'exp_{experiment}_{day}_{month}_{hour}_{minute}')

        # Check if the directory already exists
        if not os.path.exists(workspace_folder):
            # Create the directory
            os.makedirs(workspace_folder)
        with open(workspace_folder + '/distance.txt', 'w') as distance_file:
            distance_file.write(str(travelled_distance_main))
        distance_file.close()
        images_folder = os.path.join(settings['path'], directory_name)
        run_colmap(colmap_folder, workspace_folder, str(images_folder))

        for route, count_group in zip(route_by_group, range(len(route_by_group))):
            filename = settings['filename']
            directory_name = (settings['directory name'] +
                              f'_exp_{experiment}_group_{count_group}_{day}_{month}_{hour}_{minute}')
            quadcopter_control_direct_points(copp.sim,
                                             copp.client,
                                             copp.handles[settings['vision sensor names']],
                                             route,
                                             filename,
                                             directory_name)

            workspace_folder = os.path.join(settings['workspace folder'],
                                            f'exp_{experiment}_{day}_{month}_{hour}_{minute}_group_{count_group}')

            # Check if the directory already exists
            if not os.path.exists(workspace_folder):
                # Create the directory
                os.makedirs(workspace_folder)
            travelled_distance_main = 0
            for i in range(route.shape[0]):
                for j in range(i+1, route.shape[0]):
                    travelled_distance_main += np.linalg.norm(route[i] - route[j])

            with open(workspace_folder + '/distance.txt', 'w') as distance_file:
                distance_file.write(str(travelled_distance_main))
            distance_file.close()
            images_folder = os.path.join(settings['path'], directory_name)
            run_colmap(colmap_folder, workspace_folder, str(images_folder))

    copp.sim.stopSimulation()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
