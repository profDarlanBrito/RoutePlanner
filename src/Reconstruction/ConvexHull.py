import os
import pickle
import platform
import subprocess
import sys
from random import sample
from typing import Any

import numpy as np
import pyvista as pv
from numpy import dtype, float_, floating, ndarray
from numpy._typing import _64Bit
from scipy.spatial import ConvexHull

import Config
from CoppeliaInterface import CoppeliaInterface, initializations
from GeometryOperations import (
    compute_central_hemisphere_area,
    draw_cylinder_with_hemisphere,
    euler_angles_from_normal,
    find_normal_vector,
    get_geometric_objects_cell,
    get_side_hemisphere_area,
    is_line_through_convex_hull,
    points_along_line,
)
from IO import write_OP_file_3d, write_problem_file_3d
from Reconstruction.Roberts import compute_points_weights, draw_cylinders_with_hemispheres, generate_target_view_points

settings = Config.Settings.get()


def compute_edge_weight_matrix(S_cewm: dict, targets_points_of_view_cewm: dict[Any, ndarray]) -> ndarray:
    print("Starting computing distance matrix")

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
                        np.deg2rad(pt1[3:]) - np.deg2rad(pt2[3:])
                    )
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
    print(f"size of edge matrix: {edge_weight_matrix_cewm.shape[0]} x {edge_weight_matrix_cewm.shape[0]}")
    return edge_weight_matrix_cewm


def draw_cylinders_hemispheres(centroid_points_pf: dict, radius_pf: dict, target_points_pf: dict) -> tuple[
    dict[Any, ndarray[Any, dtype[Any]] | ndarray[Any, dtype[floating[_64Bit] | float_]]],
    dict[Any, list[float] | list[Any]],
    list[ndarray[Any, dtype[Any]]],
]:
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
    max_route_radius = float(settings["max route radius"])
    points_per_unit = float(settings["points per unit"])
    points_per_sphere = float(settings["points per sphere"])
    n_resolution = int(settings["n resolution"])

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
        meshes = draw_cylinder_with_hemisphere(
            plotter, cy_direction, cy_hight, n_resolution, r_mesh, centroid_points_pf[target], 0.0
        )
        cylinder = meshes["cylinder"]["mesh"]
        if not is_included_first_group:
            vector_points_pf[target] = np.array([-2.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            vector_points_weight_pf[target] = [0.0]
            conversion_table.append(np.array([-2.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
            is_included_first_group = True
        else:
            vector_points_pf[target] = np.empty([0, 6])
            vector_points_weight_pf[target] = []
        count_hemisphere = 0
        route_radius_dch = int(np.fix(max_route_radius / points_per_unit))
        weights = (route_radius_dch - 1) * [0.0]
        for cell in get_geometric_objects_cell(cylinder):
            hemisphere_radius = meshes["hemispheres"][count_hemisphere]["radius"]  #
            pos_cell = cell.center
            points_cell = cell.points[:3]
            norm_vec = find_normal_vector(*points_cell)
            yaw, pitch, roll = euler_angles_from_normal(-norm_vec)
            for k in range(1, route_radius_dch):
                camera_distance = ((points_per_sphere * k) + 1) * hemisphere_radius
                point_position = pos_cell + camera_distance * norm_vec
                if (
                    count_hemisphere == 0
                    or count_hemisphere == n_resolution
                    or count_hemisphere == cylinder.n_cells - n_resolution
                ):
                    spherical_area_dc, reach_maximum, frustum_planes, cam_pos = compute_central_hemisphere_area(
                        norm_vec,
                        pos_cell,
                        hemisphere_radius,
                        camera_distance,
                        plotter,
                        float(settings["perspective angle"]),
                        near_clip_ccha=float(settings["near clip"]),
                        far_clip_ccha=float(settings["far clip"]),
                    )
                    area = get_side_hemisphere_area(
                        cylinder.n_cells, meshes, frustum_planes, count_hemisphere, n_resolution
                    )
                    weight = spherical_area_dc + area
                    weights[k - 1] = weight
                else:
                    weight = weights[k - 1]
                vector_points_pf[target] = np.row_stack(
                    (vector_points_pf[target], np.concatenate((point_position, np.array([yaw, pitch, roll]))))
                )
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


def subgroup_formation(
    targets_border_sf: dict,
    points_of_view_contribution_sf: dict,
    target_points_of_view_sf: dict,
    positions: dict = None,
) -> tuple[dict, int]:
    """
    Forms the subgroups of points of view around each object.
    :param positions:
    :param targets_border_sf: Dictionary with the convex hull computed by Delaunay function of Scipy for each target object
    :param points_of_view_contribution_sf: Dictionary with a reward of each point of view. Each key of the dictionary is an object target
    :param target_points_of_view_sf: Dictionary with the positions of each point of view around each target object
    :return S: Dictionary with subgroup. Each key is a target object
    :return length: Total number of subgroups
    """
    print("Starting subgroup formation")

    CA_max = float(settings["CA_max"])
    max_visits = int(settings["max visits"])
    max_iter = int(settings["max iter"])
    search_size = 20  # Size of the random points that will be used to search the next position of the UAV.
    number_of_line_points = 10  # The number of the points that will be used to define a line that will be verified if is through the convex hull

    S = {}
    contribution = 0
    subgroup_idx = 0
    is_first_target = True
    cont_target = 0
    length = 0
    # Get the points of view for each target object
    for target, points in target_points_of_view_sf.items():
        if positions is None:
            object_area = 1.0
        else:
            object_area = ConvexHull(positions[target]).volume
            
        CA_max_sf = object_area * CA_max
        S[target] = []
        # Create a subgroup 0 with position and orientation equals to zero. This subgroup is the start and end subgroup
        if subgroup_idx == 0:
            S[target].append([])
            S[target][-1].append((subgroup_idx, subgroup_idx, 0, 0, 0.0, 0.0, 0, 0))
            subgroup_idx += 1
        visits_to_position = np.zeros(
            points.shape[0]
        )  # Number of visits to a point of view. Used to determine the maximum number of visits to a point of view
        if is_first_target:
            visits_to_position[0] += max_visits + 1
        indexes_of_ini_points = list(range(points.shape[0]))
        random_points = sample(
            indexes_of_ini_points, len(indexes_of_ini_points)
        )  # Selects randomly the index of points to form the groups

        show_number_of_points = 0
        for i in random_points:
            if len(S) == 1 and i == 0:
                continue

            CA = 0
            total_distance = 0
            S[target].append([])
            prior_idx = i
            max_idx = -1
            idx_list = [i]
            show_number_of_points += 1
            iteration = 0
            while CA < CA_max_sf and iteration < max_iter:
                iteration += 1
                indexes_of_points = np.random.randint(
                    low=0, high=points.shape[0], size=search_size
                )  # Select randomly the index of points where the drone can go
                max_contribution = 0
                for index in indexes_of_points:
                    if index in idx_list:
                        continue
                    distance_p2p = np.linalg.norm(
                        target_points_of_view_sf[target][prior_idx, :3] - target_points_of_view_sf[target][index, :3]
                    )
                    contribution = abs(abs(points_of_view_contribution_sf[target][index]) - distance_p2p)  #
                    distance_orientation = np.linalg.norm(
                        target_points_of_view_sf[target][prior_idx, 3:] - target_points_of_view_sf[target][index, 3:]
                    )
                    if distance_orientation < 1:
                        contribution = contribution - 0.9 * contribution
                    if contribution > max_contribution:
                        max_idx = index
                        max_contribution = abs(points_of_view_contribution_sf[target][index])

                if max_idx == -1:
                    continue

                if visits_to_position[max_idx] > max_visits:
                    continue
                is_line_through_convex_hull_sf = False
                for target_compare, hull_sf in targets_border_sf.items():
                    line_points = points_along_line(
                        target_points_of_view_sf[target][prior_idx, :3],
                        target_points_of_view_sf[target][max_idx, :3],
                        number_of_line_points,
                    )
                    is_line_through_convex_hull_sf = is_line_through_convex_hull(hull_sf, line_points)
                    if is_line_through_convex_hull_sf:
                        break
                if is_line_through_convex_hull_sf:
                    continue

                # Find duplicate point if max_idx is in the idx_list
                if max_idx in idx_list:
                    continue

                # Ignore idx equal zero on first target
                if max_idx == 0 and len(S) == 1:
                    continue

                idx_list.append(max_idx)
                distance_p2p = np.linalg.norm(
                    target_points_of_view_sf[target][prior_idx, :3] - target_points_of_view_sf[target][max_idx, :3]
                )
                total_distance = distance_p2p + total_distance
                CA += contribution

                # CA can not be greater than CA_max_sf
                if CA >= CA_max_sf:
                    break

                visits_to_position[max_idx] += 1
                prior_idx_s = length + prior_idx
                max_idx_s = length + max_idx
                # Dictionary with subgroups by object target. Each target has n subgroups. And each subgroup has your elements which is composed by a tuple with:
                S[target][-1].append(
                    (
                        subgroup_idx,  # General index of the group considering all subgroups of all objects
                        prior_idx,  # Index of the previous visited point of view. This index is by target
                        max_idx,  # Index of the next visited point of view. This index is by target
                        distance_p2p,  # Euclidean distance between the start and end points.
                        total_distance,
                        # Total travelled distance until the point max_idx. The last element of the subgroup will have the total travelled distance in subgroup
                        CA,
                        # Total reward of the subgroup until the max_idx point. The last element of the subgroup will have the total reward for the subgroup
                        prior_idx_s,  # Index of the previous visited point of view. This index considering all target
                        max_idx_s,
                    )
                )  # Index of the next visited point of view. This index is considering all target
                prior_idx = max_idx
            # If the above step do not reach the CA minimum shows a message to user.
            # if iteration >= max_iter - 1:
            #     print('Decrease CA_max')
            #     print(f'{CA=}')
            #     print(f'{len(S[target][-1])=}')
            # If the subgroup is empty remove it from the subgroup list

            # if len(S[target][-1]) == 0 or CA < settings["CA_min"]:
            if len(S[target][-1]) == 0 or CA < CA_max_sf / 2:
                S[target].pop()
            else:
                subgroup_idx += 1

        length += len(S[target])  # Compute the total length of the subgroups
        is_first_target = False
        cont_target += 1
        print(f"{target=} has {len(S[target])=} groups")
    return S, length


def execute_script(name_cops_file: str) -> None:
    try:

        if platform.system() == "Windows":
            tabu_search_exec = "tabu_search.bat"

        if platform.system() == "Linux":
            tabu_search_exec = "tabu_search.sh"

        # Execute the script using subprocess
        process = subprocess.Popen(
            [
                os.path.join(settings["COPS path"], tabu_search_exec),
                settings["python"],
                settings["COPS path"],
                os.path.join(settings["COPS dataset"], name_cops_file),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        return process
    except Exception as e:
        print("An error occurred:", e)
        return None


def print_process(process):
    # Wait for the process to finish
    stdout, stderr = process.communicate()

    # Check if there were any errors
    if process.returncode != 0:
        print("Error executing script:")
        print(stderr.decode("utf-8", errors="ignore"))
    else:
        print("Script executed successfully.")
        print(stdout.decode("utf-8", errors="ignore"))


def load_variables():

    if len(sys.argv) >= 7:
        settings["points per unit"] = float(sys.argv[2])
        settings["T_max"] = int(sys.argv[3])
        settings["number of trials"] = int(sys.argv[4])
        settings["CA_max"] = int(sys.argv[5])
        settings["obj_file"] = sys.argv[6]

    settings["save path"] = os.path.abspath(settings["save path"])

    save_path = settings["save path"]

    path = settings["path"]
    COPS_dataset = settings["COPS dataset"]
    COPS_result = settings["COPS result"]
    workspace_folder = settings["workspace folder"]

    settings["path"] = os.path.join(save_path, path)
    settings["COPS dataset"] = os.path.join(save_path, COPS_dataset)
    settings["COPS result"] = os.path.join(save_path, COPS_result)
    settings["workspace folder"] = os.path.join(save_path, workspace_folder)


def convex_hull(experiment: int):
    print("Starting convex hull ...")
    load_variables()

    if settings["use obj"] == 1:
        with open(os.path.join(settings["obj folder"], settings["obj_file"]), "rb") as file:
            positions = pickle.load(file)
            target_hull = pickle.load(file)
            centroid_points = pickle.load(file)
            radius = pickle.load(file)
            settings["object names"] = pickle.load(file)

    else:
        coppelia = CoppeliaInterface(settings)
        positions, target_hull, centroid_points, radius = initializations(coppelia)
        coppelia.sim.stopSimulation()
        del coppelia

    # COPS
    targets_points_of_view, points_of_view_contribution, conversion_table_cops = draw_cylinders_hemispheres(
        centroid_points, radius, positions
    )

    S, subgroup_size = subgroup_formation(target_hull, points_of_view_contribution, targets_points_of_view, positions)
    name_cops_file = settings["COPS problem"] + str(experiment)

    offset_table = {}
    for key, value in targets_points_of_view.items():
        offset_table[key] = len(value)

    interval_cops = {}
    offset = 0
    for key, value in targets_points_of_view.items():

        n = len(value)
        if all(value[0] == conversion_table_cops[offset]) and all(value[-1] == conversion_table_cops[offset + n - 1]):
            interval_cops[key] = [offset, offset + n - 1]
            offset += n

    write_problem_file_3d(settings["COPS dataset"],
        name_cops_file,
        conversion_table_cops,
        offset_table,
        len(settings["object names"]),
        S,
        subgroup_size,
    )

    return

    # Roberts
    plotter, target_meshes = draw_cylinders_with_hemispheres(centroid_points, radius, positions)
    target_view_points, conversion_table_op = generate_target_view_points(target_meshes, radius)
    target_view_weights = compute_points_weights(target_view_points, target_meshes, plotter)
    name_op_file = settings["OP problem"] + str(experiment)

    interval_op = {}
    offset = 0
    for key, value in target_view_points.items():

        n = len(value)
        if all(value[0] == conversion_table_op[offset]) and all(value[-1] == conversion_table_op[offset + n - 1]):
            interval_op[key] = [offset, offset + n - 1]
            offset += n

    write_OP_file_3d(
        settings["COPS dataset"], name_op_file, conversion_table_op, target_view_weights, target_view_points
    )

    print("Executing COPS ...")
    print("Executing OP ...")

    with open(os.path.join(settings["save path"], f"variables/convex_hull_{experiment}.var"), "wb") as file:
        pickle.dump(S, file)
        pickle.dump(targets_points_of_view, file)
        pickle.dump(centroid_points, file)
        pickle.dump(radius, file)
        pickle.dump(conversion_table_cops, file)
        pickle.dump(conversion_table_op, file)
        pickle.dump(interval_cops, file)
        pickle.dump(interval_op, file)

    process_cops = execute_script(name_cops_file)
    process_op = execute_script(name_op_file)

    print_process(process_cops)
    print_process(process_op)

    print("Ending convex hull")
