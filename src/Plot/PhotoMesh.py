import os

import numpy as np
import open3d as o3d

# Carregar o mesh
file_name = "meshed-poisson-crop.ply"
# file_name = "meshed-poisson.ply"
base_path = "/mnt/e/Arquivos/wsl_arquivos/experiments_artigo/dalek/build/0/COLMAP_results/exp_0_21_11_8_29_group_Old_Build/dense/0"
# base_path = "/mnt/e/Arquivos/wsl_arquivos/experiments_artigo/wall-e/build/0/COLMAP_results/exp_5_21_11_9_4_group_Old_Build/dense/0"
mesh = o3d.io.read_triangle_mesh(os.path.join(base_path, file_name))

# path = "/mnt/c/Users/thgsa/OneDrive/Documentos/Scene/mesh/bob.obj"
# mesh = o3d.io.read_triangle_mesh(path)

# Definir a posição da câmera e o ponto de interesse
cam_height = 2
camera_position = np.array([-5, -8, cam_height])  # Posição da câmera no espaço
look_at_point = np.array([-12, -12, cam_height])  # Ponto que a câmera deve olhar

# Configurar o renderizador offscreen
width, height = 1920, 1080  # Tamanho da imagem
renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

material = o3d.visualization.rendering.MaterialRecord()

# Adicionar o mesh à cena
scene = renderer.scene
scene.add_geometry(f"mesh_0", mesh, material)

# path = "/mnt/c/Users/thgsa/OneDrive/Documentos/Scene/mesh/cars/"
# for i, file in enumerate(os.listdir(path)):
#     if file.endswith(".obj"):
#         mesh = o3d.io.read_triangle_mesh(os.path.join(path, file), enable_post_processing=True)
#         scene.add_geometry(f"mesh_{i}", mesh, material)


sun_dir = np.zeros((3, 1))
scene.set_lighting(scene.LightingProfile.DARK_SHADOWS, sun_dir)
up_vector = np.array([0, 0, 1])  # Eixo 'up'
renderer.setup_camera(
    vertical_field_of_view=60, center=look_at_point, eye=camera_position, up=up_vector, near_clip=0, far_clip=100
)

# Renderizar a imagem e salvar
image = renderer.render_to_image()
o3d.io.write_image("Old_Build_mesh_good.png", image)
# o3d.io.write_image("Old_Build_mesh_bad.png", image)

# camera_position = np.array([7.5, -5.0, 2.0])  # Posição da câmera no espaço
# renderer.setup_camera(
#     vertical_field_of_view=60, center=look_at_point, eye=camera_position, up=up_vector, near_clip=0, far_clip=100
# )
# image = renderer.render_to_image()
# o3d.io.write_image("bobcat_mesh_good_test_2.png", image)


print("Imagem salva com sucesso!")
