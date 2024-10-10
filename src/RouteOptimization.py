import ast
import csv
import datetime
import multiprocessing
import os
import pickle
import platform
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from random import sample
from typing import Any, Dict, List, Tuple

import cv2 as cv
import numpy as np
import open3d as o3d
import pyvista as pv
from numpy import bool_, dtype, float_, floating, ndarray
from numpy._typing import _64Bit
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation as Rot

from MeshAnalysis import mesh_analysis
import config
from Colmap.colmap import run_colmap_program, statistics_colmap
from Colmap.pipeline import generate_mesh_poisson
from ConvexHull.ConvexHull import convex_hull
from Coppelia.CoppeliaInterface import CoppeliaInterface
from MathUtil.GeometryOperations import (compute_central_hemisphere_area,
                                         draw_cylinder_with_hemisphere,
                                         euler_angles_from_normal,
                                         intersect_plane_sphere,
                                         point_between_planes)
from PointCloud.PointCloud import generate_poisson_mesh, point_cloud
from ViewPoint.ViewPoint import view_point

settings = config.Settings.get()


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
        cy_hight = float(settings['height proportion']) * np.max(target_points_pf[target][:, 2])
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


def update_current_experiment(value_stage: int) -> None:
    with open(os.path.join(settings['save path'], '.progress'), 'wb') as file:
        pickle.dump(value_stage, file)


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