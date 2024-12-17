import os
import platform

import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation as Rot

import Config
from Colmap.pipeline import *
from Database import COLMAPDatabase

settings = Config.Settings.get()


def run_colmap_program(colmap_folder: str, workspace_folder: str, images_folder: str) -> None:
    if platform.system() == "Windows":
        run_colmap(os.path.join(colmap_folder, "COLMAP.bat"), workspace_folder, str(images_folder))

    if platform.system() == "Linux":
        run_colmap("colmap", workspace_folder, images_folder)


def camera_file(db, model_folder):
    # https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h

    camera = "cameras.txt"
    with open(os.path.join(model_folder, camera), "w") as camera_file:
        camera_param = settings["camera param"]
        camera_id = 1
        camera_file.write(
            f"{camera_id} PINHOLE {camera_param[0]} {camera_param[1]} {camera_param[2]} {camera_param[3]} {camera_param[4]} {camera_param[5]}\n"
        )
    
    camera_model = 1 # PINHOLE
    camera_param = settings["camera param"]
    width_height = camera_param[:2]
    params = camera_param[2:]
    db.add_camera(camera_model, width_height[0], width_height[1], params, camera_id=camera_id)

    return camera_id


def images_file(db, model_folder, camera_id, image_folder):
    ref_file = "ref_images.txt"
    with open(os.path.join(image_folder, ref_file), "r") as ref_file:
        lines = ref_file.readlines()

    lines = [l.split() for l in lines]
    lines.sort(key=lambda l: l[0])  # sort by the image name

    images = "images.txt"
    with open(os.path.join(model_folder, images), "w") as images_file:
        for i, (name, x, y, z, qw, qx, qy, qz) in enumerate(lines):
            
            R = Rot.from_quat([float(qx), float(qy), float(qz), float(qw)]).as_matrix().T
            c = np.array([float(x), float(y), float(z)])

            R_180 = np.diag([-1, -1, 1]) @ R
            t = -R_180 @ c
            tx, ty, tz = t

            qx, qy, qz, qw = Rot.from_matrix(R_180).as_quat()

            image_id = i + 1
            images_file.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {name}\n\n")
            db.add_image(name, camera_id, image_id, qw, qx, qy, qz, tx, ty, tz)


def points_file(model_folder):

    points3D = "points3D.txt"
    with open(os.path.join(model_folder, points3D), "w") as points_file:
        points_file.write("")


def create_base_files(workspace_folder, image_folder, sparse_dir):

    database_path = os.path.join(workspace_folder, "database.db")
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    model_folder = os.path.join(sparse_dir, "model")
    os.mkdir(model_folder)

    camera_id = camera_file(db, model_folder)

    images_file(db, model_folder, camera_id, image_folder)

    points_file(model_folder)

    db.commit()
    db.close()


def sparse_reconstruction(colmap_exec: str, workspace_folder: str, image_folder: str):
    # Create sparse folder
    sparse_dir = os.path.join(workspace_folder, "sparse").replace("\\", "/")
    os.mkdir(sparse_dir)

    database_creator(colmap_exec, workspace_folder)

    create_base_files(workspace_folder, image_folder, sparse_dir)

    extract_features(colmap_exec, workspace_folder, image_folder)

    perform_exhaustive_matching(colmap_exec, workspace_folder)

    point_triangulator(colmap_exec, workspace_folder, image_folder, sparse_dir)

    return sparse_dir


def dense_reconstruction(colmap_exec: str, workspace_folder: str, image_folder: str, sparse_dir: str):
    if settings["dense model"] == 1:
        # Create dense folder
        dense_dir = os.path.join(workspace_folder, "dense").replace("\\", "/")
        os.mkdir(dense_dir)

    for folder in os.listdir(sparse_dir):

        if settings["dense model"] == 0:
            break

        sub_dense_dir = os.path.join(dense_dir, folder).replace("\\", "/")
        sub_dense_sparse_dir = os.path.join(sub_dense_dir, "sparse")
        sub_sparse_dir = os.path.join(sparse_dir, folder).replace("\\", "/")
        os.mkdir(sub_dense_dir)

        undistort_images(colmap_exec, workspace_folder, image_folder, sub_sparse_dir, sub_dense_dir)

        perform_stereo_matching(colmap_exec, workspace_folder, sub_dense_dir)

        perform_stereo_fusion(colmap_exec, workspace_folder, sub_dense_dir)

        convert_model(colmap_exec, workspace_folder, sub_dense_sparse_dir)


def run_colmap(colmap_exec: str, workspace_folder: str, image_folder: str) -> None:
    """
    Execute the COLMAP script on Windows
    :param colmap_folder: Folder where is stored the COLMAP.bat file
    :param workspace_folder: Folder where the COLMAP results will be stored
    :param image_folder: Folder to images used for reconstruction. There is no name pattern to images
    :return: Nothing
    """
    print("Executing colmap script ...")
    try:
        sparse_dir = sparse_reconstruction(colmap_exec, workspace_folder, image_folder)

        dense_reconstruction(colmap_exec, workspace_folder, image_folder, sparse_dir)

        print("Script executed successfully.")
    except Exception as e:
        print("An error occurred:", e)


def statistics_colmap(colmap_folder_sc, workspace_folder_sc, MNRE_array=np.empty(0)) -> ndarray | None:
    print("Creating colmap statistics.")
    i = 0
    try:
        while True:
            statistic_folder = os.path.join(workspace_folder_sc, f"sparse/{i}/")
            if os.path.exists(statistic_folder):
                # exec_name = ''
                if platform.system() == "Windows":
                    colmap_exec = os.path.join(colmap_folder_sc, "COLMAP.bat")
                if platform.system() == "Linux":
                    colmap_exec = "colmap"

                with subprocess.Popen(
                    [colmap_exec, "model_analyzer", "--path", statistic_folder],
                    shell=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                ) as process:
                    output, stderr = process.communicate(
                        timeout=10
                    )  # Capture stdout and stderr. The result is shown on stderr

                # Check if there were any errors
                if process.returncode != 0:
                    print("Error executing script:")
                    print(stderr)
                else:
                    if output is not None:
                        print(output)
                    if stderr is None:
                        print("model_analyzer do not create any data")
                        return
                    else:
                        print(stderr)
                        # Read data from COLMAP output
                        points_value_idx = stderr.find(":", stderr.find("Points")) + 1
                        number_of_points = int(
                            stderr[points_value_idx : stderr.find("\n", points_value_idx)]
                        )  # Number of points
                        error_value_idx = stderr.find(":", stderr.find("Mean reprojection error")) + 1
                        error_value = float(
                            stderr[error_value_idx : stderr.find("p", error_value_idx)]
                        )  # Reconstruction error
                        MNRE = error_value / number_of_points  # Compute de Mean Normalized Reconstruction Error

                        # Save important data to file
                        with open(statistic_folder + "MNRE.txt", "w") as statistic_file:
                            statistic_file.write(f"MNRE: {MNRE}\n")
                            statistic_file.write(f"Mean reprojection error: {error_value}\n")
                            statistic_file.write(f"Points: {number_of_points}")

                        with open(statistic_folder + "stat.txt", "w") as stat_file:
                            stat_file.write(stderr)

                        MNRE_array = np.concatenate((MNRE_array, [MNRE]))
                    print("COLMAP data model analyzer executed successfully.")
            else:
                break
            # statistic_file.close()
            # stat_file.close()
            i += 1
        return MNRE_array
    except Exception as e:
        print("An error occurred:", e)
        return None
