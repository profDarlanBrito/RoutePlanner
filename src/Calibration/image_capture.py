import os

import cv2 as cv
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from GeometryOperations import get_rotation_quat


def generate_points(num_points, bounding_box):
    """
    Gera pontos aleatórios dentro de um bounding box em R3.

    Parâmetros:
    - num_points: Número de pontos a serem gerados.
    - bounding_box: Uma tupla contendo os limites do bounding box no formato:
      ((x_min, x_max), (y_min, y_max), (z_min, z_max))

    Retorna:
    - Uma matriz numpy de shape (num_points, 3) contendo os pontos gerados.
    """
    x_min, x_max = bounding_box[0]
    y_min, y_max = bounding_box[1]
    z_min, z_max = bounding_box[2]

    points = np.empty((num_points, 3))
    points[:, 0] = np.random.uniform(x_min, x_max, num_points)
    points[:, 1] = np.random.uniform(y_min, y_max, num_points)
    points[:, 2] = np.random.uniform(z_min, z_max, num_points)

    return points


def generate_camera_positions(num_positions):
    current = generate_points(num_positions, ((0.5, 1.5), (-1, 1), (0.01, 1)))
    target = generate_points(num_positions, ((0.1, 0.1), (-0.3, 0.3), (0.0, 0.2)))

    pos = [p.tolist() for p in current]
    ori = [get_rotation_quat(curr_pos, target_pos).tolist() for curr_pos, target_pos in zip(current, target)]

    return pos, ori


def get_image(sim, sequence: int, vision_handle: int, directory_name: str):
    img, resolution = sim.getVisionSensorImg(vision_handle)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)

    image_name = f"ca_img_{sequence}.png"

    image_path = os.path.join(directory_name, image_name)

    img = cv.flip(cv.cvtColor(img, cv.COLOR_BGR2RGB), 0)

    cv.imwrite(image_path, img)


num_images = 50
count_image = 0

client = RemoteAPIClient()
sim = client.getObject("sim")
sim.startSimulation()
sensor_handle = sim.getObject("./Vision_sensor")

dir_name = "calibration_images/"
os.makedirs(dir_name, exist_ok=True)


for pos, ori in zip(*generate_camera_positions(num_images)):

    sim.setObjectPosition(sensor_handle, pos, sim.handle_world)
    sim.setObjectQuaternion(sensor_handle, ori, sim.handle_world)

    total_time = sim.getSimulationTime() + 0.02
    while sim.getSimulationTime() < total_time:
        client.step()

    get_image(sim, count_image, sensor_handle, dir_name)
    count_image += 1

sim.stopSimulation()
