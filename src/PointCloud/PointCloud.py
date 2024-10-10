import platform
import shutil
import config
import os
import pickle
import numpy as np

from Colmap.colmap import run_colmap_program, statistics_colmap
from Colmap.pipeline import generate_mesh_poisson

settings = config.Settings.get()

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
