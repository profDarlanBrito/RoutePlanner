import os
import numpy as np
import open3d as o3d
import csv
from scipy.spatial.transform import Rotation as Rot
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed


def read_camera_pos_files(file_path, ref_file_path):

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

            real_camera_center = list(map(float, line_split[1:]))

            if image_name in camera_centers:
                camera_centers[image_name].append(np.array(real_camera_center))

            line = f.readline()

    return camera_centers


def affine_transformation(P, Q):
    """minimize ||S - Qx||"""

    m, q = np.shape(P)
    p, n = np.shape(Q)

    if not (m == p and q == n == 3):
        raise RuntimeError("Invalid input matrix")

    S = np.hstack((P, np.ones((m, 1))))
    S_plus = np.linalg.pinv(S)
    x = S_plus @ Q

    R = x[:-1].T
    t = x[-1].T

    T = np.identity(q + 1)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def horn(P, Q):
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
    source_points = np.array(source_cloud.points)
    target_points = np.array(target_cloud.points)

    return horn(source_points, target_points)


def mesh_lab_hausdorff_distance(ms, sample=1_000_000, i=1, j=0):
    d1 = ms.get_hausdorff_distance(sampledmesh=i, targetmesh=j, samplenum=sample)
    d2 = ms.get_hausdorff_distance(sampledmesh=j, targetmesh=i, samplenum=sample)
    return d1, d2


def hausdorff(source_points, target_points):
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

        if np.isnan(curr_min):
            print("FIND NAN VALUE")
            break

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


def hausdorff_distance(source_pcd, target_pcd):
    source = np.array(source_pcd.points)
    target = np.array(target_pcd.points)

    source = source[~np.isnan(source).any(axis=1)]
    target = target[~np.isnan(target).any(axis=1)]

    print(f"Hausdorff PointCloud with {len(source)} points")
    print(f"Hausdorff PointCloud with {len(target)} points")

    params = [(source, target), (target, source)]
    return Pool().starmap(hausdorff, params)


def get_last_directories(path, count):
    path_parts = path.split(os.sep)

    if path_parts[-1] == '':
        path_parts.pop()
    
    last_three_dirs = os.path.join(*path_parts[-count:])
    
    return last_three_dirs


def save_to_csv(data, filename):
    keys = data[0].keys()
    with open(filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)

def main():
    list_image_path = [
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_200_CA_min_100/3D_Reconstruction/Images/scene_builds_exp_0_group_0_12_7_9_45",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_200_CA_min_100/3D_Reconstruction/Images/scene_builds_exp_0_group_1_12_7_9_45",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_200_CA_min_100/3D_Reconstruction/Images/scene_builds_exp_0_group_2_12_7_9_45",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_200_CA_min_100/3D_Reconstruction/Images/scene_builds_exp_0_group_3_12_7_9_45",

        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_200_CA_min_100/3D_Reconstruction/Images/scene_builds_spriral_exp_0_group_0_12_7_9_45",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_200_CA_min_100/3D_Reconstruction/Images/scene_builds_spriral_exp_0_group_1_12_7_9_45",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_200_CA_min_100/3D_Reconstruction/Images/scene_builds_spriral_exp_0_group_2_12_7_9_45",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_200_CA_min_100/3D_Reconstruction/Images/scene_builds_spriral_exp_0_group_3_12_7_9_45",

        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_400_CA_min_300/3D_Reconstruction/Images/scene_builds_exp_0_group_0_12_7_10_1",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_400_CA_min_300/3D_Reconstruction/Images/scene_builds_exp_0_group_1_12_7_10_1",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_400_CA_min_300/3D_Reconstruction/Images/scene_builds_exp_0_group_2_12_7_10_1",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_400_CA_min_300/3D_Reconstruction/Images/scene_builds_exp_0_group_3_12_7_10_1",

        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_400_CA_min_300/3D_Reconstruction/Images/scene_builds_spriral_exp_0_group_0_12_7_10_1",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_400_CA_min_300/3D_Reconstruction/Images/scene_builds_spriral_exp_0_group_1_12_7_10_1",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_400_CA_min_300/3D_Reconstruction/Images/scene_builds_spriral_exp_0_group_2_12_7_10_1",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_400_CA_min_300/3D_Reconstruction/Images/scene_builds_spriral_exp_0_group_3_12_7_10_1",
 
    ]

    list_reconstruction_path = [
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_200_CA_min_100/COLMAP_results/exp_0_12_7_9_45_group_0/dense/0/",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_200_CA_min_100/COLMAP_results/exp_0_12_7_9_45_group_1/dense/0",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_200_CA_min_100/COLMAP_results/exp_0_12_7_9_45_group_2/dense/0",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_200_CA_min_100/COLMAP_results/exp_0_12_7_9_45_group_3/dense/1",

        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_200_CA_min_100/COLMAP_results/spiral_exp_0_12_7_9_45_group_0/dense/0",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_200_CA_min_100/COLMAP_results/spiral_exp_0_12_7_9_45_group_1/dense/0",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_200_CA_min_100/COLMAP_results/spiral_exp_0_12_7_9_45_group_2/dense/0",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_200_CA_min_100/COLMAP_results/spiral_exp_0_12_7_9_45_group_3/dense/1",

        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_400_CA_min_300/COLMAP_results/exp_0_12_7_10_1_group_0/dense/0",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_400_CA_min_300/COLMAP_results/exp_0_12_7_10_1_group_1/dense/0",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_400_CA_min_300/COLMAP_results/exp_0_12_7_10_1_group_2/dense/0",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_400_CA_min_300/COLMAP_results/exp_0_12_7_10_1_group_3/dense/0",

        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_400_CA_min_300/COLMAP_results/spiral_exp_0_12_7_10_1_group_0/dense/1",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_400_CA_min_300/COLMAP_results/spiral_exp_0_12_7_10_1_group_1/dense/1",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_400_CA_min_300/COLMAP_results/spiral_exp_0_12_7_10_1_group_2/dense/0",
        "/home/clust3rr/disco_E/Build_Hausdorff/CA_max_400_CA_min_300/COLMAP_results/spiral_exp_0_12_7_10_1_group_3/dense/0",
    ]

    list_plt_path = [
        "mesh_obj_2/Build/West Building/west_build_sample.ply",
        "mesh_obj_2/Build/Chinese Old Wall Building/chinese_old_build_sample.ply",
        "mesh_obj_2/Build/Old Building/old_build_sample.ply",
        "mesh_obj_2/Build/CasaVelha/casa_velha_sample.ply",

        "mesh_obj_2/Build/CasaVelha/casa_velha_sample.ply",
        "mesh_obj_2/Build/Chinese Old Wall Building/chinese_old_build_sample.ply",
        "mesh_obj_2/Build/Old Building/old_build_sample.ply",
        "mesh_obj_2/Build/West Building/west_build_sample.ply",

        "mesh_obj_2/Build/West Building/west_build_sample.ply",
        "mesh_obj_2/Build/Old Building/old_build_sample.ply",
        "mesh_obj_2/Build/Chinese Old Wall Building/chinese_old_build_sample.ply",
        "mesh_obj_2/Build/CasaVelha/casa_velha_sample.ply",

        "mesh_obj_2/Build/CasaVelha/casa_velha_sample.ply",
        "mesh_obj_2/Build/Chinese Old Wall Building/chinese_old_build_sample.ply",
        "mesh_obj_2/Build/Old Building/old_build_sample.ply",
        "mesh_obj_2/Build/West Building/west_build_sample.ply",
    ]

    data_flatten = []
    data_delta = []
    data_sum = []

    for image_path, reconstruction_path, plt_path in zip(list_image_path, list_reconstruction_path, list_plt_path):

        sparce_folder = "sparse"
        file_name = "images.txt"
        ref_file_name = "ref_images.txt"
        mesh_file = "meshed-poisson.ply"

        sparce_path = os.path.join(reconstruction_path, sparce_folder)
        file_path = os.path.join(sparce_path, file_name)
        ref_file_path = os.path.join(image_path, ref_file_name)
        mesh_plt_path = os.path.join(reconstruction_path, mesh_file)


        camera_centers = read_camera_pos_files(file_path, ref_file_path)

        cc = []
        rcc = []

        for pos in camera_centers.values():
            cc.append(pos[0])
            rcc.append(pos[1])

        # Converter para PointCloud do Open3D
        camera_cloud = o3d.geometry.PointCloud()
        camera_cloud.points = o3d.utility.Vector3dVector(cc)

        real_camera_cloud = o3d.geometry.PointCloud()
        real_camera_cloud.points = o3d.utility.Vector3dVector(rcc)

        camera_cloud.paint_uniform_color([1, 0, 0])  # Vermelho para o source
        real_camera_cloud.paint_uniform_color([0, 1, 0])  # Verde para o target

        transformation = align_points(camera_cloud, real_camera_cloud)
        camera_cloud.transform(transformation)

        n_error = np.linalg.norm(np.array(camera_cloud.points) - np.array(real_camera_cloud.points)) / len(
            np.array(camera_cloud.points)
        )
        print(f"{n_error=:.8f}")

        T = transformation
        while n_error > 0.0005:
            n = len(camera_cloud.points)
            colmap_points = np.array(camera_cloud.points)
            real_points = np.array(real_camera_cloud.points)

            mfind = 0
            k = -1
            min = np.inf
            for i in range(n):
                v = np.linalg.norm(colmap_points[i] - real_points[i])
                if v > mfind and v > 0.2:
                    mfind = v
                    k = i
                    
                if v < min:
                    min = v

            if not n > 2:
                raise Exception("Can not find the transformation matrix")

            camera_cloud.transform(np.linalg.inv(T))
            colmap_points = np.array(camera_cloud.points)
            real_points = np.array(real_camera_cloud.points)

            if k == -1:
                break

            colmap_points = np.delete(colmap_points, k , 0)
            real_points = np.delete(real_points, k , 0)

            camera_cloud.points = o3d.utility.Vector3dVector(colmap_points)
            real_camera_cloud.points = o3d.utility.Vector3dVector(real_points)

            T = horn(colmap_points, real_points)

            camera_cloud.transform(T)

            n_error = np.linalg.norm(np.array(colmap_points) - np.array(real_points)) / n
            print(f"{n_error=}")

        camera_cloud.paint_uniform_color([1, 0, 0])  # Vermelho para o source
        real_camera_cloud.paint_uniform_color([0, 1, 0])  # Verde para o target

        source_pcd = o3d.io.read_point_cloud(plt_path)
        target_pcd = o3d.io.read_point_cloud(mesh_plt_path)

        print(source_pcd)
        print(target_pcd)

        # o3d.visualization.draw_geometries([source_pcd, target_pcd])

        target_pcd.transform(T)

        # o3d.visualization.draw_geometries([source_pcd, target_pcd])

        bbp = source_pcd.get_axis_aligned_bounding_box()
        bbp.color = (1, 0, 0)
        bounding_box_points = np.array(bbp.get_box_points())
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
        n = len(target_pcd_points)
        bounding_points = []
        for p in target_pcd_points:
            if p[0] < b_min[0] - threshold or p[1] < b_min[1] - threshold or p[2] < b_min[2] - threshold:
                continue
            elif p[0] > b_max[0] + threshold or p[1] > b_max[1] + threshold or p[2] > b_max[2] + threshold:
                continue
            else:
                bounding_points.append(p)
        
        target_pcd.points = o3d.utility.Vector3dVector(bounding_points)

        print("Used Transformation Matrix:")
        print(T)

        new_mesh_path =  os.path.dirname(mesh_plt_path)
        o3d.io.write_point_cloud(os.path.join(new_mesh_path, 'meshed-poisson-crop.ply'), target_pcd)

        print("Start calculate hausdorff_distance")
        # hausdorff_dist = hausdorff_distance(source_pcd, target_pcd)
        # print(get_last_directories(reconstruction_path, 5))
        # print(f'{hausdorff_dist=}')

        # data_sum.append({
        #     'reconstruction_path': get_last_directories(reconstruction_path, 5),
        #     'ply': os.path.basename(plt_path), 
        #     'source_points': hausdorff_dist[0]['source_points'],
        #     'min': np.min((hausdorff_dist[0]['min'], hausdorff_dist[1]['min'])),
        #     'max': np.max((hausdorff_dist[0]['max'], hausdorff_dist[1]['max'])),
        #     'mae': hausdorff_dist[0]['mae'] + hausdorff_dist[1]['mae'],
        #     'rmse': hausdorff_dist[0]['rmse'] + hausdorff_dist[1]['rmse'],
        # })

        # data_delta.append({
        #     'reconstruction_path': get_last_directories(reconstruction_path, 5),
        #     'plt': os.path.basename(plt_path), 
        #     'source_points': hausdorff_dist[0]['source_points'],
        #     'min': np.abs(hausdorff_dist[0]['min'] - hausdorff_dist[1]['min']),
        #     'max': np.abs(hausdorff_dist[0]['max'] - hausdorff_dist[1]['max']),
        #     'mae': np.abs(hausdorff_dist[0]['mae'] - hausdorff_dist[1]['mae']),
        #     'rmse': np.abs(hausdorff_dist[0]['rmse'] - hausdorff_dist[1]['rmse']),
        # })
        
        # data_flatten.append({
        #     'reconstruction_path': get_last_directories(reconstruction_path, 5),
        #     'plt': os.path.basename(plt_path), 
        #     'points': hausdorff_dist[0]['source_points'],
        #     'min': hausdorff_dist[0]['min'],
        #     'max': hausdorff_dist[0]['max'],
        #     'mae': hausdorff_dist[0]['mae'],
        #     'rmse': hausdorff_dist[0]['rmse'],
        # })

        # data_flatten.append({
        #     'reconstruction_path': get_last_directories(reconstruction_path, 5),
        #     'plt': os.path.basename(plt_path), 
        #     'points': hausdorff_dist[1]['source_points'],
        #     'min': hausdorff_dist[1]['min'],
        #     'max': hausdorff_dist[1]['max'],
        #     'mae': hausdorff_dist[1]['mae'],
        #     'rmse': hausdorff_dist[1]['rmse'],
        # })
        
        # Visualizar as nuvens de pontos alinhadas
        source_pcd.paint_uniform_color([0, 1, 0])  # Verde para o source
        target_pcd.paint_uniform_color([1, 0, 0])  # Vermelho para o source_sample
        o3d.visualization.draw_geometries([source_pcd, target_pcd, bbp])

    # save_to_csv(data_delta, 'data_delta.csv')
    # save_to_csv(data_flatten, 'data_flatten.csv')
    # save_to_csv(data_sum, 'data_sum.csv')

if __name__ == "__main__":
    main()
