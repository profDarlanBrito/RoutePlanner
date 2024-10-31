import os
import shutil
import subprocess

feature_extractor_file = "config/feature_extractor.ini"
exhaustive_file = "config/exhaustive_matcher.ini"
mapper_file = "config/mapper.ini"
point_triangulator_file = "config/point_triangulator.ini"
image_undistorter_file = "config/image_undistorter.ini"
patch_match_stereo_file = "config/patch_match_stereo.ini"
stereo_fusion_file = "config/stereo_fusion.ini"
poisson_mesher_file = "config/poisson_mesher.ini"
model_aligner_file = "config/model_aligner.ini"
model_converter_file = "config/model_converter.ini"


def write_config_file(config_file_name: str, workspace_folder: str, config_lines: list[str]):
    config_path = os.path.join(workspace_folder, os.path.basename(config_file_name)).replace("\\", "/")
    with open(config_path, "w") as config_file:
        config_file.writelines(config_lines)
    return config_path


def execute_colmap_command(colmap_exec: str, command: str, config_file_path: str):
    process = subprocess.Popen([colmap_exec, command, "--project_path", config_file_path])
    process.communicate()  # Wait for the process to finish


def database_creator(colmap_exec, workspace_folder):
    database_path = os.path.join(workspace_folder, "database.db")
    process = subprocess.Popen([colmap_exec, "database_creator", "--database_path", database_path])
    process.communicate()  # Wait for the process to finish


def extract_features(colmap_exec, workspace_folder, image_folder):
    # Extract features
    with open(feature_extractor_file, "r") as feature_config_file_read:
        feature_extractor_config_str = feature_config_file_read.readlines()
        feature_extractor_config_str[3] = f"database_path={workspace_folder}/database.db\n"
        feature_extractor_config_str[4] = f"image_path={image_folder}\n"

    feature_config_path = write_config_file(feature_extractor_file, workspace_folder, feature_extractor_config_str)
    execute_colmap_command(colmap_exec, "feature_extractor", feature_config_path)


def perform_exhaustive_matching(colmap_exec, workspace_folder):
    # Perform exhaustive matching
    with open(exhaustive_file, "r") as exhaustive_matcher_file_read:
        exhaustive_config_str = exhaustive_matcher_file_read.readlines()
        exhaustive_config_str[3] = f"database_path={workspace_folder}/database.db\n"

    exhaustive_matcher_config_path = write_config_file(exhaustive_file, workspace_folder, exhaustive_config_str)
    execute_colmap_command(colmap_exec, "exhaustive_matcher", exhaustive_matcher_config_path)


def point_triangulator(colmap_exec, workspace_folder, image_folder, sparse_dir):

    model_folder = os.path.join(sparse_dir, "model")
    model_output_folder = os.path.join(sparse_dir, "0")

    os.mkdir(model_output_folder)

    # Run the point_triangulator
    with open(point_triangulator_file, "r") as point_triangulator_file_read:
        point_tri_config_str = point_triangulator_file_read.readlines()
        point_tri_config_str[3] = f"database_path={workspace_folder}/database.db\n"
        point_tri_config_str[4] = f"image_path={image_folder}\n"
        point_tri_config_str[5] = f"input_path={model_folder}\n"
        point_tri_config_str[6] = f"output_path={model_output_folder}\n"

    point_triangulator_config_path = write_config_file(point_triangulator_file, workspace_folder, point_tri_config_str)
    execute_colmap_command(colmap_exec, "point_triangulator", point_triangulator_config_path)
    shutil.rmtree(model_folder)


def run_mapper(colmap_exec, workspace_folder, image_folder, sparse_dir):
    # Run the mapper
    with open(mapper_file, "r") as mapper_file_read:
        mapper_config_str = mapper_file_read.readlines()
        mapper_config_str[3] = f"database_path={workspace_folder}/database.db\n"
        mapper_config_str[4] = f"image_path={image_folder}\n"
        mapper_config_str[5] = f"output_path={sparse_dir}\n"

    mapper_config_path = write_config_file(mapper_file, workspace_folder, mapper_config_str)
    execute_colmap_command(colmap_exec, "mapper", mapper_config_path)


def align_scene(colmap_exec, workspace_folder, input_path, output_path, ref_images_file_path):
    # Register camera centers
    with open(model_aligner_file, "r") as model_aligner_file_read:
        model_aligner_config_str = model_aligner_file_read.readlines()
        model_aligner_config_str[3] = f"input_path={input_path}\n"
        model_aligner_config_str[4] = f"output_path={output_path}\n"
        model_aligner_config_str[5] = f"ref_images_path={ref_images_file_path}\n"

    model_aligner_config_path = write_config_file(model_aligner_file, workspace_folder, model_aligner_config_str)
    execute_colmap_command(colmap_exec, "model_aligner", model_aligner_config_path)


def undistort_images(colmap_exec, workspace_folder, image_folder, sub_sparse_dir, sub_dense_dir):
    # Undistort images
    with open(image_undistorter_file, "r") as image_undistorter_file_read:
        image_config_str = image_undistorter_file_read.readlines()
        image_config_str[0] = f"image_path={image_folder}\n"
        image_config_str[1] = f"input_path={sub_sparse_dir}\n"
        image_config_str[2] = f"output_path={sub_dense_dir}\n"

    image_undistorter_config_path = write_config_file(image_undistorter_file, workspace_folder, image_config_str)
    execute_colmap_command(colmap_exec, "image_undistorter", image_undistorter_config_path)


def perform_stereo_matching(colmap_exec, workspace_folder, sub_dense_dir):
    # Perform stereo matching
    with open(patch_match_stereo_file, "r") as patch_match_stereo_file_read:
        stereo_config_str = patch_match_stereo_file_read.readlines()
        stereo_config_str[3] = f"workspace_path={sub_dense_dir}\n"

    patch_match_stereo_config_path = write_config_file(patch_match_stereo_file, workspace_folder, stereo_config_str)
    execute_colmap_command(colmap_exec, "patch_match_stereo", patch_match_stereo_config_path)


def perform_stereo_fusion(colmap_exec, workspace_folder, sub_dense_dir):
    # Perform stereo fusion
    with open(stereo_fusion_file, "r") as stereo_fusion_file_read:
        stereo_fusion_config_str = stereo_fusion_file_read.readlines()
        stereo_fusion_config_str[3] = f"workspace_path={sub_dense_dir}\n"
        stereo_fusion_config_str[6] = f"output_path={sub_dense_dir}/fused.ply\n"

    stereo_fusion_config_path = write_config_file(stereo_fusion_file, workspace_folder, stereo_fusion_config_str)
    execute_colmap_command(colmap_exec, "stereo_fusion", stereo_fusion_config_path)


def generate_mesh_poisson(colmap_exec, workspace_folder, sub_dense_dir):
    # Generate mesh using Poisson meshing
    with open(poisson_mesher_file, "r") as poisson_mesher_file_read:
        poisson_config_str = poisson_mesher_file_read.readlines()
        poisson_config_str[3] = f"input_path={sub_dense_dir}/fused.ply\n"
        poisson_config_str[4] = f"output_path={sub_dense_dir}/meshed-poisson.ply\n"

    poisson_mesher_config_path = write_config_file(poisson_mesher_file, workspace_folder, poisson_config_str)
    execute_colmap_command(colmap_exec, "poisson_mesher", poisson_mesher_config_path)


def convert_model(colmap_exec, workspace_folder, target_dir):
    # Converte bin model to txt model
    with open(model_converter_file, "r") as model_converter_file_read:
        model_converter_config_str = model_converter_file_read.readlines()
        model_converter_config_str[4] = f"input_path={target_dir}\n"
        model_converter_config_str[5] = f"output_path={target_dir}\n"

    model_converter_config_path = write_config_file(model_converter_file, workspace_folder, model_converter_config_str)
    execute_colmap_command(colmap_exec, "model_converter", model_converter_config_path)


def align_scene_poses(colmap_exec, workspace_folder, image_folder, sparse_dir):
    ref_images_file_path = os.path.join(image_folder, "ref_images.txt")
    if os.path.isfile(ref_images_file_path):
        # Create sparse aligned folder
        sparse_aligned_dir = os.path.join(workspace_folder, "sparse_aligned").replace("\\", "/")
        os.mkdir(sparse_aligned_dir)

        for folder in os.listdir(sparse_dir):
            input_path = os.path.join(sparse_dir, folder).replace("\\", "/")
            output_path = os.path.join(sparse_aligned_dir, folder).replace("\\", "/")
            os.mkdir(output_path)

            align_scene(colmap_exec, workspace_folder, input_path, output_path, ref_images_file_path)

        sparse_dir = sparse_aligned_dir

    return sparse_dir
