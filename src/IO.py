import ast
import csv
import os
import pickle
import shutil

import numpy as np
from numpy import ndarray

import Config

settings = Config.Settings.get()
fieldnames = [
    "NAME: ",
    "TYPE: ",
    "COMMENT: ",
    "DIMENSION: ",
    "TMAX: ",
    "START_CLUSTER: ",
    "END_CLUSTER: ",
    "CLUSTERS: ",
    "SUBGROUPS: ",
    "EDGE_WEIGHT_TYPE: ",
    "NODE_COORD_SECTION: ",
    "GTSP_SUBGROUP_SECTION: ",
    "GTSP_CLUSTER_SECTION: ",
]


def ConvertArray2String(fileCA2S, array: ndarray):
    np.set_printoptions(threshold=10000000000)
    np.savetxt(fileCA2S, array, fmt="%.3f", delimiter=" ")
    return fileCA2S


def copy_file(source_path, destination_path):
    try:
        # Copy the file from source_path to destination_path
        shutil.copy(source_path, destination_path)
        print(f"File copied successfully from {source_path} to {destination_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def read_route_csv_file(
    file_path, S_rrcf: dict, targets_points_of_vew_rrcf: dict, experiment: int
) -> tuple[ndarray, float, list[ndarray]]:
    route_rrcf = np.empty([0, 6])
    route_by_group = [np.empty([0, 6])] * len(settings["object names"])
    travelled_distance = 0
    route_by_object = {}
    try:
        with open(file_path, newline="", encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=";")
            for row in csv_reader:
                route_reward = row[5]
                route_str = row[8]
    except Exception as e:
        print(f"An error occurred: {e}")

    with open(os.path.join(settings["save path"], f"route_reward_file_{experiment}.txt"), "w") as csvfile:
        route_reward = route_reward.replace(",", ".")
        csvfile.write("route_reward: " + route_reward)

    chose_subgroups = ast.literal_eval(route_str.replace("  ", ","))
    bigger_idx = 0
    table_rrcf = []
    for target_rrcf, points_rrcf in targets_points_of_vew_rrcf.items():
        table_rrcf.append([target_rrcf, bigger_idx, bigger_idx + points_rrcf.shape[0]])
        bigger_idx += points_rrcf.shape[0] + 1
    count_group = 0
    is_group_zero = True
    is_group_zero_zero = True
    group_name = "0"
    for S_idx_rrcf in chose_subgroups:
        for information_rrcf in table_rrcf:
            # if information_rrcf[1] <= S_idx_rrcf <= information_rrcf[2]:
            is_first_element = True
            for group_rrcf in S_rrcf[information_rrcf[0]]:
                # print(f'Group size: {len(group_rrcf)}')
                for element in group_rrcf:
                    if element[0] == S_idx_rrcf:
                        # print(f'Size of selected group: {len(group_rrcf)}')
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
                                group_name = information_rrcf[0]
                                route_by_group[count_group] = np.row_stack(
                                    (route_by_group[count_group], pt_prior_coordinates, pt_post_coordinates)
                                )
                        else:
                            route_rrcf = np.row_stack((route_rrcf, pt_post_coordinates))
                            route_by_group[count_group] = np.row_stack(
                                (route_by_group[count_group], pt_post_coordinates)
                            )
        # route_by_object[information_rrcf[0]] = route_by_group[count_group]
        if is_group_zero:
            is_group_zero = False
        else:
            route_by_object[group_name] = route_by_group[count_group]
            count_group += 1
    print(f"{travelled_distance=}")
    route_rrcf = np.row_stack((route_rrcf, route_rrcf[0]))
    return route_rrcf, travelled_distance, route_by_group, route_by_object


def write_problem_file(
    dir_wpf: str,
    filename_wpf: str,
    edge_weight_matrix_wpf: ndarray,
    number_of_targets: int,
    S_wpf: dict,
    subgroup_size_wpf: int,
):
    print("Starting writing problem file")
    subgroup_count = 0
    write_bin_problem_file(dir_wpf, filename_wpf, edge_weight_matrix_wpf)
    # Create the directory
    os.makedirs(dir_wpf, exist_ok=True)

    complete_file_name = dir_wpf + filename_wpf + ".cops"
    # print(f'{complete_file_name=}')
    GTSP_CLUSTER_SECTION_str = []
    with open(complete_file_name, "w") as copsfile:
        for field_wpf in fieldnames:
            if field_wpf == "NAME: ":
                copsfile.write(field_wpf + filename_wpf + settings["directory name"] + "\n")
            elif field_wpf == "TYPE: ":
                copsfile.write(field_wpf + "TSP\n")
            elif field_wpf == "COMMENT: ":
                copsfile.write(field_wpf + "Optimization for reconstruction\n")
            elif field_wpf == "DIMENSION: ":
                copsfile.write(field_wpf + str(edge_weight_matrix_wpf.shape[0]) + "\n")
            elif field_wpf == "TMAX: ":
                copsfile.write(field_wpf + str(settings["T_max"]) + "\n")
            elif field_wpf == "START_CLUSTER: ":
                copsfile.write(field_wpf + "0\n")
            elif field_wpf == "END_CLUSTER: ":
                copsfile.write(field_wpf + "0\n")
            elif field_wpf == "CLUSTERS: ":
                copsfile.write(field_wpf + str(number_of_targets) + "\n")
            elif field_wpf == "SUBGROUPS: ":
                copsfile.write(field_wpf + str(subgroup_size_wpf) + "\n")
            elif field_wpf == "DUBINS_RADIUS: ":
                copsfile.write(field_wpf + "50" + "\n")
            elif field_wpf == "EDGE_WEIGHT_TYPE: ":
                copsfile.write(field_wpf + "EXPLICIT" + "\n")
            elif field_wpf == "EDGE_WEIGHT_FORMAT: ":
                copsfile.write(field_wpf + "FULL_MATRIX" + "\n")
            elif field_wpf == "EDGE_WEIGHT_SECTION":
                copsfile.close()
                with open(complete_file_name, "a") as copsfile:
                    copsfile.write(field_wpf + "\n")
                    ConvertArray2String(copsfile, edge_weight_matrix_wpf)
            elif field_wpf == "GTSP_SUBGROUP_SECTION: ":
                with open(complete_file_name, "a") as copsfile:
                    copsfile.write(f"{field_wpf}cluster_id cluster_profit id-vertex-list\n")
                    count_cluster = 0
                    GTSP_CLUSTER_SECTION_str = [[]] * len(settings["object names"])
                    for target_wpf, S_spf in S_wpf.items():
                        GTSP_CLUSTER_SECTION_str[count_cluster] = [[]] * (len(S_spf) + 1)
                        GTSP_CLUSTER_SECTION_str[count_cluster][0] = f"{count_cluster} "
                        count_idx = 1
                        for lS_spf in S_spf:
                            copsfile.write(
                                f"{lS_spf[0][0]} {lS_spf[-1][5]} {lS_spf[0][6]} "
                                + " ".join(str(vertex[7]) for vertex in lS_spf)
                                + "\n"
                            )
                            GTSP_CLUSTER_SECTION_str[count_cluster][count_idx] = f"{lS_spf[0][0]} "
                            count_idx += 1
                        count_cluster += 1
            elif field_wpf == "GTSP_CLUSTER_SECTION: ":
                with open(complete_file_name, "a") as copsfile:
                    copsfile.write(f"{field_wpf} set_id id-cluster-list\n")
                    for cluster_idxs in GTSP_CLUSTER_SECTION_str:
                        copsfile.writelines(cluster_idxs)
                        copsfile.write("\n")

    copsfile.close()


def write_bin_problem_file(
    dir_wbpf: str, filename_wbpf: str, edge_weight_matrix_wbpf: ndarray
):  # , number_of_targets: int,
    # S_wbpf: dict, subgroup_size_wbpf: int):
    print("Writing binary file")

    os.makedirs(dir_wbpf, exist_ok=True)

    complete_file_name = dir_wbpf + filename_wbpf + "_b.cops"

    data = ("NAME", filename_wbpf + settings["directory name"], "EDGE_WEIGHT_SECTION", edge_weight_matrix_wbpf)

    # Write the data to a binary file
    with open(complete_file_name, "wb") as file:
        pickle.dump(data, file)


def write_OP_file_3d(
    dir_wpf: str, filename_wpf: str, node_coord: list, points_of_view_contribution: dict, targets_points_of_view: dict
):
    print("Starting writing problem file")
    subgroup_count = 0
    # Create the directory
    os.makedirs(dir_wpf, exist_ok=True)

    complete_file_name = dir_wpf + filename_wpf + ".cops"
    # print(f'{complete_file_name=}')
    GTSP_CLUSTER_SECTION_str = []
    with open(complete_file_name, "w") as copsfile:
        for field_wpf in fieldnames:
            if field_wpf == "NAME: ":
                copsfile.write(field_wpf + filename_wpf + settings["directory name"] + "\n")
            elif field_wpf == "TYPE: ":
                copsfile.write(field_wpf + "TSP\n")
            elif field_wpf == "COMMENT: ":
                copsfile.write(field_wpf + "Optimization for reconstruction\n")
            elif field_wpf == "DIMENSION: ":
                copsfile.write(field_wpf + str(len(node_coord)) + "\n")
            elif field_wpf == "TMAX: ":
                copsfile.write(field_wpf + str(settings["T_max"]) + "\n")
            elif field_wpf == "START_CLUSTER: ":
                copsfile.write(field_wpf + "0\n")
            elif field_wpf == "END_CLUSTER: ":
                copsfile.write(field_wpf + "0\n")
            elif field_wpf == "CLUSTERS: ":
                copsfile.write(field_wpf + str(len(node_coord)) + "\n")
            elif field_wpf == "SUBGROUPS: ":
                copsfile.write(field_wpf + str(len(node_coord)) + "\n")
            elif field_wpf == "EDGE_WEIGHT_TYPE: ":
                copsfile.write(field_wpf + "EUC_3D" + "\n")
            elif field_wpf == "NODE_COORD_SECTION: ":
                copsfile.write("NODE_COORD_SECTION: id_vertex x y z\n")
                for i, point in enumerate(node_coord):
                    copsfile.write(f"{i} {point[0]} {point[1]} {point[2]}\n")
            elif field_wpf == "GTSP_SUBGROUP_SECTION: ":
                copsfile.write(f"{field_wpf}cluster_id cluster_profit id-vertex-list\n")

                id = 0
                for _, contribution_list in points_of_view_contribution.items():
                    for i, contribution in enumerate(contribution_list):
                        copsfile.write(f"{id + i} {contribution} {id + i}\n")

                    id += i + 1

            elif field_wpf == "GTSP_CLUSTER_SECTION: ":
                copsfile.write(f"{field_wpf} set_id id-cluster-list\n")
                id = 0
                for _, contribution_list in points_of_view_contribution.items():
                    for i in range(len(contribution_list)):
                        copsfile.write(f"{id + i} {id + i}\n")

                    id += i + 1

    copsfile.close()


def write_problem_file_3d(
    dir_wpf: str,
    filename_wpf: str,
    node_coord: list,
    offset_table: dict,
    number_of_targets: int,
    S_wpf: dict,
    subgroup_size_wpf: int,
):
    print("Starting writing problem file")
    subgroup_count = 0
    # Create the directory
    os.makedirs(dir_wpf, exist_ok=True)

    complete_file_name = dir_wpf + filename_wpf + ".cops"
    # print(f'{complete_file_name=}')
    GTSP_CLUSTER_SECTION_str = []
    with open(complete_file_name, "w") as copsfile:
        for field_wpf in fieldnames:
            if field_wpf == "NAME: ":
                copsfile.write(field_wpf + filename_wpf + settings["directory name"] + "\n")
            elif field_wpf == "TYPE: ":
                copsfile.write(field_wpf + "TSP\n")
            elif field_wpf == "COMMENT: ":
                copsfile.write(field_wpf + "Optimization for reconstruction\n")
            elif field_wpf == "DIMENSION: ":
                copsfile.write(field_wpf + str(len(node_coord)) + "\n")
            elif field_wpf == "TMAX: ":
                copsfile.write(field_wpf + str(settings["T_max"]) + "\n")
            elif field_wpf == "START_CLUSTER: ":
                copsfile.write(field_wpf + "0\n")
            elif field_wpf == "END_CLUSTER: ":
                copsfile.write(field_wpf + "0\n")
            elif field_wpf == "CLUSTERS: ":
                copsfile.write(field_wpf + str(number_of_targets + 1) + "\n")
            elif field_wpf == "SUBGROUPS: ":
                copsfile.write(field_wpf + str(subgroup_size_wpf) + "\n")
            elif field_wpf == "EDGE_WEIGHT_TYPE: ":
                copsfile.write(field_wpf + "EUC_3D" + "\n")
            elif field_wpf == "NODE_COORD_SECTION: ":
                copsfile.write("NODE_COORD_SECTION: id_vertex x y z\n")
                for i, point in enumerate(node_coord):
                    copsfile.write(f"{i} {point[0]} {point[1]} {point[2]}\n")
            elif field_wpf == "GTSP_SUBGROUP_SECTION: ":
                copsfile.write(f"{field_wpf}cluster_id cluster_profit id-vertex-list\n")
                count_cluster = 0
                GTSP_CLUSTER_SECTION_str = [[]] * (len(settings["object names"]) + 1)

                offset = 0
                for target_wpf, S_spf in S_wpf.items():

                    if len(S_spf[0]) == 1 and S_spf[0][0][0] == 0:
                        GTSP_CLUSTER_SECTION_str[count_cluster] = ["0 ", "0 "]
                        count_cluster += 1

                        copsfile.write(
                            f"{S_spf[0][0][0]} {S_spf[0][-1][5]} "
                            + " ".join(str(vertex[1] + offset) for vertex in S_spf[0])
                            + "\n"
                        )

                        S_spf = S_spf[1:]

                    GTSP_CLUSTER_SECTION_str[count_cluster] = [[]] * (len(S_spf) + 1)
                    GTSP_CLUSTER_SECTION_str[count_cluster][0] = f"{count_cluster} "
                    count_idx = 1
                    for lS_spf in S_spf:
                        copsfile.write(f"{lS_spf[0][0]} {lS_spf[-1][5]} " + str(lS_spf[0][1] + offset) + " ")
                        copsfile.write(" ".join(str(vertex[2] + offset) for vertex in lS_spf) + "\n")

                        GTSP_CLUSTER_SECTION_str[count_cluster][count_idx] = f"{lS_spf[0][0]} "
                        count_idx += 1
                    count_cluster += 1

                    offset += offset_table[target_wpf]
            elif field_wpf == "GTSP_CLUSTER_SECTION: ":
                copsfile.write(f"{field_wpf} set_id id-cluster-list\n")
                for cluster_idxs in GTSP_CLUSTER_SECTION_str:
                    copsfile.writelines(cluster_idxs)
                    copsfile.write("\n")

    copsfile.close()
