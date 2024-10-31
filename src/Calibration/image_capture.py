import os

import cv2 as cv
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from scipy.spatial.transform import Rotation as Rot


def get_rotation_quat(curr_pos, target_pos):
    """
    Calculates the quaternion representing the rotation needed to align the current position
    to face the target position.

    The function computes a "look-at" vector from the current position to the target position,
    then calculates the corresponding rotation matrix. The matrix is converted into a quaternion
    to represent the 3D rotation.

    Parameters:
    ----------
    curr_pos : array-like
        The current position as a 3D vector (x, y, z).

    target_pos : array-like
        The target position as a 3D vector (x, y, z).

    Returns:
    --------
    quaternion : np.ndarray
        A 4-element array representing the rotation as a quaternion [x, y, z, w].

    Notes:
    ------
    - The "look-at" vector is normalized to get the direction from the current position
      to the target.
    - The "up" vector is assumed to be [0, 0, 1], which is aligned with the Z-axis.
    - The right and new up vectors are calculated via cross products to form an orthogonal
      coordinate system, which is then used to create the rotation matrix.
    - The `Rot.from_matrix()` function from `scipy.spatial.transform` is used to convert the
      rotation matrix into a quaternion.
    """
    look_at = np.array(target_pos) - np.array(curr_pos)
    look_at = look_at / np.linalg.norm(look_at)

    up = np.array([0.0, 0.0, 1.0])

    right = np.cross(up, look_at)
    right = right / np.linalg.norm(right)

    new_up = np.cross(look_at, right)

    rotation_matrix = np.array([right, new_up, look_at]).T

    rotation = Rot.from_matrix(rotation_matrix)

    return rotation.as_quat()


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
