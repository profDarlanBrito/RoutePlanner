import csv
import os
import pickle
from multiprocessing import Pool

import numpy as np
import open3d as o3d
from numpy import ndarray
from scipy.spatial.transform import Rotation as Rot

import Config
from GeometryOperations import horn

settings = Config.Settings.get()


def read_camera_pos_files(file_path: str, ref_file_path: str) -> dict:
    """Read the camera position files and return a dictionary with the camera positions"""

    # Read the file and extract only the camera centers
    camera_centers = {}

    with open(file_path, "r") as f:
        # Skip the first four lines
        for _ in range(4):
            f.readline()

        line = f.readline()
        while line:
            line_split = line.replace("\n", "").split(" ")
            image_name = line_split[-1]
            qw, qx, qy, qz = map(float, line_split[1:5])
            tx, ty, tz = map(float, line_split[5:8])

            # Convert quaternion to rotation matrix
            rotation = Rot.from_quat([qx, qy, qz, qw])
            R = rotation.as_matrix()

            # Calculate the camera center
            t = np.array([tx, ty, tz])
            camera_center = -R.T @ t
            camera_centers[image_name] = [camera_center]

            # Read the empty line and the next data line
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


def align_points(source_cloud, target_cloud):
    """Align two point clouds using Horn's method"""
    source_points = np.array(source_cloud.points)
    target_points = np.array(target_cloud.points)

    return horn(source_points, target_points)


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


def metrics(source_points: ndarray, target_points: ndarray) -> dict:
    """Calculate the metrics between two point clouds"""
    n = len(source_points)
    min = np.inf
    max = -np.inf
    mae = 0
    rmse = 0

    if len(target_points) == 0 or n == 0:
        return {"source_points": n, "min": min, "max": max, "mae": -np.inf, "rmse": -np.inf}

    for point in source_points:
        dist = np.linalg.norm(target_points - point, axis=1)
        curr_min = np.min(dist)

        if curr_min < 0.05:
            curr_min = 0

        if curr_min < min:
            min = curr_min

        if curr_min > max:
            max = curr_min

        mae += curr_min
        rmse += curr_min**2

    mae /= n
    rmse = np.sqrt(rmse / n)

    return {"source_points": n, "min": min, "max": max, "mae": mae, "rmse": rmse}


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

    if path_parts[-1] == "":
        path_parts.pop()

    last_three_dirs = os.path.join(*path_parts[-count:])

    return last_three_dirs


def back_directories(path, count):
    """Navigate back through the last directories of a path"""
    path_parts = path.split(os.sep)

    if path_parts[-1] == "":
        path_parts.pop()

    n = len(path_parts)

    if path_parts[0] == "":
        path_parts[0] = os.sep

    last_three_dirs = os.path.join(*path_parts[: n - count])

    return last_three_dirs


def get_experiment_number(reconstruction_path: str):
    path = back_directories(reconstruction_path, 2)
    path = get_last_directories(path, 1)

    split_name = path.split("_")

    if split_name[0] == "spiral":
        return split_name[2]

    return split_name[1]


def save_to_csv(data, filename):
    """Save the data to a CSV file"""
    keys = data[0].keys()
    with open(filename, "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()

        for line in data:
            if line is None:
                continue

            dict_writer.writerow(line)


def process_reconstruction(image_path, reconstruction_path, plt_path):
    """Process the reconstruction and calculate the metrics"""
    sparce_folder = "sparse"
    file_name = "images.txt"
    ref_file_name = "ref_images.txt"
    mesh_file = "meshed-poisson.ply"
    point_cloud_file = "fused.ply"

    sparce_path = os.path.join(reconstruction_path, sparce_folder)
    file_path = os.path.join(sparce_path, file_name)
    ref_file_path = os.path.join(image_path, ref_file_name)
    mesh_plt_path = os.path.join(reconstruction_path, mesh_file)
    point_cloud_plt_path = os.path.join(reconstruction_path, point_cloud_file)

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
    target_pcd = o3d.io.read_point_cloud(point_cloud_plt_path)

    target_pcd.transform(T)

    bbp = source_pcd.get_axis_aligned_bounding_box()
    bounding_points = crop_point_cloud(target_pcd, bbp)

    cropped_mesh = target_mesh.crop(bbp)

    target_pcd.points = o3d.utility.Vector3dVector(bounding_points)

    new_mesh_path = os.path.dirname(mesh_plt_path)

    o3d.io.write_point_cloud(os.path.join(new_mesh_path, "fused-crop.ply"), target_pcd)
    o3d.io.write_triangle_mesh(os.path.join(new_mesh_path, "meshed-poisson-crop.ply"), cropped_mesh)

    print("Start calculate hausdorff_distance")
    metrics_dist = calculate_metrics_distance(source_pcd, target_pcd)
    last_dir = get_last_directories(reconstruction_path, 3)
    print(last_dir)

    reconstruction_path = os.path.relpath(reconstruction_path)
    distance_file_path = back_directories(reconstruction_path, 2)
    distance_file_path = os.path.join(distance_file_path, "distance.txt")

    with open(distance_file_path, "r") as file:
        dist = float(file.readline().strip())

    experiment = get_experiment_number(reconstruction_path)

    if last_dir.startswith("spiral") or last_dir.startswith("op") or last_dir.startswith("random"):
        route_reward = "none"

    else:
        results_cops_file = f"{settings['COPS problem']}{experiment}.csv"
        with open(os.path.join(settings["COPS result"], results_cops_file), "r") as csv_reward_file:
            csv_reader = csv.DictReader(csv_reward_file, delimiter=";")

            line = next(csv_reader)
            route_reward = line["profit"]
            route_reward = float(route_reward.replace(",", "."))

    return {
        "reconstruction_path": last_dir,
        "ply": os.path.basename(plt_path),
        "source_points": metrics_dist[0]["source_points"],
        "distance": dist,
        "min": np.min((metrics_dist[0]["min"], metrics_dist[1]["min"])),
        "max": np.max((metrics_dist[0]["max"], metrics_dist[1]["max"])),
        "mae": metrics_dist[0]["mae"] + metrics_dist[1]["mae"],
        "rmse": metrics_dist[0]["rmse"] + metrics_dist[1]["rmse"],
        "route_reward": route_reward,
    }


def process_paths(list_image_path, list_reconstruction_path, list_plt_path):
    data = []

    for image_path, reconstruction_path, plt_path in zip(list_image_path, list_reconstruction_path, list_plt_path):
        data.append(process_reconstruction(image_path, reconstruction_path, plt_path))

    save_to_csv(data, os.path.join(settings["save path"], "metrics.csv"))


def mesh_analysis():
    print("Initiating mesh analysis")

    obj_name = set()
    workspace_folder = settings["workspace folder"]
    for folder in os.listdir(workspace_folder):
        object_name_file = os.path.join(settings["workspace folder"], folder)
        object_name_file = os.path.join(object_name_file, "object_name.txt")

        if not os.path.isfile(object_name_file):
            continue

        with open(object_name_file, "r") as file:
            obj = file.readline().strip()
            obj_name.add(obj)

    list_image_path = []
    list_reconstruction_path = []
    list_plt_path = []

    experiment = settings["number of trials"]
    for exp in range(experiment):
        with open(os.path.join(settings["save path"], f"variables/view_point_{exp}.var"), "rb") as f:
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
            workspace_folder_group = os.path.join(
                settings["workspace folder"], f"exp_{exp}_{day}_{month}_{hour}_{minute}_group_{obj}"
            )
            op_workspace_folder_group = os.path.join(
                settings["workspace folder"], f"op_exp_{exp}_{day}_{month}_{hour}_{minute}_group_{obj}"
            )
            spiral_workspace_folder_group = os.path.join(
                settings["workspace folder"], f"spiral_exp_{exp}_{day}_{month}_{hour}_{minute}_group_{obj}"
            )
            random_workspace_folder_group = os.path.join(
                settings["workspace folder"], f"random_exp_{exp}_{day}_{month}_{hour}_{minute}_group_{obj}"
            )

            images_folder = os.path.join(
                settings["path"], f"scene_builds_exp_{exp}_group_{obj}_{day}_{month}_{hour}_{minute}"
            )
            images_folder_op = os.path.join(
                settings["path"], f"scene_builds_op_exp_{exp}_group_{obj}_{day}_{month}_{hour}_{minute}"
            )
            images_folder_spiral = os.path.join(
                settings["path"], f"scene_builds_spiral_exp_{exp}_group_{obj}_{day}_{month}_{hour}_{minute}"
            )
            images_folder_random = os.path.join(
                settings["path"], f"scene_builds_random_exp_{exp}_group_{obj}_{day}_{month}_{hour}_{minute}"
            )

            ply_path = f"mesh_obj/{obj}.ply"
            if os.path.isdir(workspace_folder_group):

                dense_folder = os.path.join(workspace_folder_group, "dense")

                if os.path.isdir(dense_folder):
                    for i in os.listdir(dense_folder):
                        reconstruction_path = os.path.join(dense_folder, i)

                        list_image_path.append(images_folder)
                        list_reconstruction_path.append(reconstruction_path)
                        list_plt_path.append(ply_path)

            if os.path.isdir(op_workspace_folder_group):

                dense_folder = os.path.join(op_workspace_folder_group, "dense")
                if os.path.isdir(dense_folder):
                    for i in os.listdir(dense_folder):
                        reconstruction_path = os.path.join(dense_folder, i)

                        list_image_path.append(images_folder_op)
                        list_reconstruction_path.append(reconstruction_path)
                        list_plt_path.append(ply_path)

            if os.path.isdir(spiral_workspace_folder_group):

                dense_folder = os.path.join(spiral_workspace_folder_group, "dense")
                if os.path.isdir(dense_folder):
                    for i in os.listdir(dense_folder):
                        reconstruction_path = os.path.join(dense_folder, i)

                        list_image_path.append(images_folder_spiral)
                        list_reconstruction_path.append(reconstruction_path)
                        list_plt_path.append(ply_path)

            if os.path.isdir(random_workspace_folder_group):

                dense_folder = os.path.join(random_workspace_folder_group, "dense")
                if os.path.isdir(dense_folder):
                    for i in os.listdir(dense_folder):
                        reconstruction_path = os.path.join(dense_folder, i)

                        list_image_path.append(images_folder_random)
                        list_reconstruction_path.append(reconstruction_path)
                        list_plt_path.append(ply_path)

    process_paths(list_image_path, list_reconstruction_path, list_plt_path)
