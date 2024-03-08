import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from CoppeliaInterface import CoppeliaInterface
import pyvista as pv
from config import parse_settings_file
from random import sample
from GeometryOperations import draw_cylinder_with_hemisphere, compute_central_hemisphere_area

number_points_view = 1500
CA_max = 50  # Bigger number the route has more points
max_route_radius = 4  # Bigger number the route increase the maximum radius of the points of view.
points_per_sphere = 1  # Density of points in the radius. If the number increase density decrease
height_proportion = 1.5  # The proportion of the tallest z height to make the cylinder
max_visits = 20  # Define the maximum number of times that the point can be visited
max_iter = 100  # Maximum number of iteration to try catch a subgroup


def create_sample_data(target_positions: dict) -> tuple:
    """
    Function to create sample data without any link to reality.
    :return targets_hull: The Delaunay data that gives the convex hull around each object
    :return targets_center: The geometric center of each object
    :return targets_points_of_view: The points of view generated out of the object
    """
    # targets_hull = {}
    targets_center = {}
    targets_points_of_view = {}
    for i in range(3):
        obj_name = f'O_{i}'
        position = 20 * np.random.rand(4, 3) - 10
        targets_center[obj_name] = np.mean(position, 0)
        targets_points_of_view[obj_name] = 20 * np.random.rand(number_points_view, 6) - 10
    return targets_center, targets_points_of_view


def camera_view_evaluation(targets_points_of_view_cve: dict):
    print('Starting creating evaluation matrix')
    points_of_view_contribution = {}
    for target, points in targets_points_of_view_cve.items():
        points_of_view_contribution[target] = 20 * np.random.rand(points.shape[0])
    return points_of_view_contribution


def is_point_inside(point, hull):
    # Check if the given point is within the convex hull
    point_in_hull = hull.find_simplex(point) >= 0
    return point_in_hull


def is_line_through_convex_hull(hull, line):
    for point in line:
        if is_point_inside(point, hull):
            return True
    return False


def points_along_line(start_point, end_point, num_points):
    # Generate num_points equally spaced between start_point and end_point
    x = np.linspace(start_point[0], end_point[0], num_points)
    y = np.linspace(start_point[1], end_point[1], num_points)
    z = np.linspace(start_point[2], end_point[2], num_points)
    points = np.column_stack((x, y, z))
    return points


def subgroup_formation(targets_border_sf: dict, points_of_view_contribution_sf: dict,
                       target_points_of_view_sf: dict, positions_sf: dict) -> dict:
    print('Starting subgroup formation')
    S = {}
    contribution = 0
    for target, points in target_points_of_view_sf.items():
        S[target] = []
        visits_to_position = np.zeros(points.shape[0])
        indexes_of_ini_points = list(range(points.shape[0]))
        random_points = sample(indexes_of_ini_points, len(indexes_of_ini_points))
        show_number_of_points = 0
        for i in random_points:
            CA = 0
            total_distance = 0
            S[target].append([])
            prior_idx = i
            max_idx = -1
            idx_list = [i]
            print(f'Point {show_number_of_points} of {len(indexes_of_ini_points)}')
            show_number_of_points += 1
            iteration = 0
            while CA < CA_max and iteration < max_iter:
                iteration += 1
                indexes_of_points = np.random.randint(low=0, high=points.shape[0], size=10)
                max_contribution = 0
                for index in indexes_of_points:
                    if index in idx_list:
                        continue
                    is_point_inside_sf = False
                    is_line_through_convex_hull_sf = False
                    for target_compare, hull_sf in targets_border_sf.items():
                        is_point_inside_sf = is_point_inside(points[index, :3], hull_sf)
                        if is_point_inside_sf:
                            break
                        if target_compare == target:
                            continue
                        line_points = points_along_line(target_points_of_view_sf[target][prior_idx, :3],
                                                        target_points_of_view_sf[target][index, :3], 100)
                        is_line_through_convex_hull_sf = is_line_through_convex_hull(hull_sf, line_points)
                        if is_line_through_convex_hull_sf:
                            break
                    if is_point_inside_sf:
                        continue
                    if is_line_through_convex_hull_sf:
                        continue
                    distance_p2p = np.linalg.norm(
                        target_points_of_view_sf[target][prior_idx, :3] - target_points_of_view_sf[target][index, :3])
                    contribution = abs(abs(points_of_view_contribution_sf[target][index]) - distance_p2p)
                    if contribution > max_contribution:
                        max_idx = index
                        max_contribution = abs(points_of_view_contribution_sf[target][index])

                if max_idx == -1:
                    continue

                if visits_to_position[max_idx] > max_visits:
                    continue
                idx_list.append(max_idx)
                distance_p2p = np.linalg.norm(
                    target_points_of_view_sf[target][prior_idx, :3] - target_points_of_view_sf[target][max_idx, :3])
                total_distance = distance_p2p + total_distance
                CA += contribution
                S[target][-1].append((prior_idx, max_idx, distance_p2p, total_distance, CA))
                visits_to_position[max_idx] += 1
                prior_idx = max_idx
            if len(S[target][-1]) == 0:
                S[target].pop()
    return S


def find_route(S_fr: dict, points_of_view_contribution_sf: dict = None, target_points_of_view_sf: dict = None):
    print('Starting finding a route ...')
    route = {}
    for target, s_fr in S_fr.items():
        CA_fr_max = -1
        Si_chose = []
        for Si_fr in S_fr[target]:
            if Si_fr[-1][-1] > CA_fr_max:
                Si_chose = Si_fr
                CA_fr_max = Si_fr[-1][-1]
        route[target] = Si_chose
    return route


def save_points(route: dict, targets_points_of_view_sr: dict):
    print('Starting saving ...')
    route_points = np.empty([0, 6])
    for target, data_s in route.items():
        for data in data_s:
            point_start = targets_points_of_view_sr[target][data[0]]
            point_end = targets_points_of_view_sr[target][data[1]]
            route_points = np.row_stack((route_points, point_end))
    np.savetxt('positions.csv', route_points, delimiter=',')


def initializations(copp) -> tuple:
    """
    Function to get the points from CoppeliaSim. The points of each object can not be at the same plane, at least one
    must be a different plane. On CoppeliaSim you must add discs around the object to form a convex hull these points
    must call Disc[0], Disc[1], ... , Disc[n]. These points must be son of a plane named O[0], O[1], ... , O[n]. These
    objects in CoppeliaSim scene must have the property Object is model on window Scene Object Properties checked. To
    access these properties you can only double-click on object.
    :return:
    """
    positions = {}
    j = 0
    targets_hull_i = {}
    centroid_points_i = {}
    radius_i = {}
    while True:
        copp.handles[f'./O[{j}]'] = copp.sim.getObject(":/O", {'index': j, 'noError': True})
        if copp.handles[f'./O[{j}]'] < 0:
            break
        positions[f'O[{j}]'] = np.empty([0, 3])
        i = 0
        while True:
            handle = copp.sim.getObject(f":/O[{j}]/Disc", {'index': i, 'noError': True})
            if handle < 0:
                break
            positions[f'O[{j}]'] = np.row_stack((positions[f'O[{j}]'],
                                                 copp.sim.getObjectPosition(handle,
                                                                            copp.sim.handle_world)))
            i += 1

        targets_hull_i[f'O[{j}]'] = Delaunay(positions[f'O[{j}]'])
        centroid_points_i[f'O[{j}]'], radius_i[f'O[{j}]'] = _centroid_poly(positions[f'O[{j}]'])
        j = j + 1

    return positions, targets_hull_i, centroid_points_i, radius_i


def _centroid_poly(poly: np.ndarray):
    T = Delaunay(poly).simplices
    n = T.shape[0]
    W = np.zeros(n)
    C = 0

    for m in range(n):
        sp = poly[T[m, :], :]
        sp += np.random.normal(0, 1e-10, sp.shape)
        W[m] = ConvexHull(sp).volume
        C += W[m] * np.mean(sp, axis=0)

    tmp_center = C / np.sum(W)
    max_distance = 0
    for m in range(n):
        sp = poly[T[m, :], :2]
        for spl in sp:
            distance = np.linalg.norm(spl - tmp_center[:2])
            if distance > max_distance:
                max_distance = distance

    return tmp_center, max_distance


def get_geometric_objects_cell(geometric_objects):
    for i in range(geometric_objects.n_cells):
        yield geometric_objects.get_cell(i)


def find_normal_vector(point1, point2, point3):
    vec1 = np.array(point2) - np.array(point1)
    vec2 = np.array(point3) - np.array(point1)
    cross_vec = np.cross(vec1, vec2)
    return cross_vec / np.linalg.norm(cross_vec)


def euler_angles_from_normal(normal_vector):
    """
    Computes Euler angles (in degrees) based on a normal vector of direction.

    Args:
    - normal_vector: A numpy array representing the normal vector of direction.

    Returns:
    - Euler angles (in degrees) as a tuple (roll, pitch, yaw).
    """
    # Normalize the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # Calculate yaw angle
    yaw = np.arctan2(normal_vector[1], normal_vector[0]) * 180 / np.pi

    # Calculate pitch angle
    pitch = np.arcsin(-normal_vector[2]) * 180 / np.pi

    # Calculate roll angle
    roll = np.arctan2(normal_vector[2], np.sqrt(normal_vector[0] ** 2 + normal_vector[1] ** 2)) * 180 / np.pi

    return roll, pitch, yaw


def compute_radial_area():
    print('Starting computing radial area')


def draw_cylinders_hemispheres(centroid_points_pf: dict, radius_pf: dict, target_points_pf: dict) -> dict:
    """
    Draw the hemispheres and the cylinders around the object
    :param centroid_points_pf: Dictionary of arrays of points with the central points of the objects
    :param radius_pf: Computed radius of the cylinders around each object
    :param target_points_pf: Computed points of the convex hull of the objects
    :return: vector_of_points: Dictionary with points around each object
    :return: Dictionary of weight of each point
    """
    print('Starting showing data')
    # Create a plotter
    plotter = pv.Plotter()
    vector_points_pf = {}
    vector_points_weight_pf = {}
    central_area_computed = False
    computed_area_by_hemisphere = []
    for target in centroid_points_pf.keys():
        cy_direction = np.array([0, 0, 1])
        n_resolution = 24
        cy_hight = height_proportion * np.max(target_points_pf[target][:, 2])
        r_mesh = radius_pf[target]
        h = np.cos(np.pi / n_resolution) * r_mesh
        # l = np.sqrt(np.abs(4 * h ** 2 - 4 * r_mesh ** 2))

        # Find the radius of the spheres
        meshes = draw_cylinder_with_hemisphere(plotter,
                                               cy_direction,
                                               cy_hight,
                                               n_resolution,
                                               r_mesh,
                                               centroid_points_pf[target])
        cylinder = meshes['cylinder']['mesh']

        vector_points_pf[target] = np.empty([0, 6])
        vector_points_weight_pf[target] = []
        hemisphere_radius = meshes['hemispheres'][0]['radius']
        for cell in get_geometric_objects_cell(cylinder):
            pos_cell = cell.center
            points_cell = cell.points[:3]
            norm_vec = find_normal_vector(*points_cell)
            roll, pitch, yaw = euler_angles_from_normal(-norm_vec)
            reach_maximum = False
            for k in range(max_route_radius):
                camera_radius = ((points_per_sphere * k) + 1) * hemisphere_radius
                point_position = pos_cell + camera_radius * norm_vec
                spherical_area_dc = 2 * np.pi * hemisphere_radius ** 2
                if not central_area_computed:
                    if not reach_maximum:
                        spherical_area_dc, reach_maximum = compute_central_hemisphere_area(norm_vec,
                                                                                           pos_cell,
                                                                                           hemisphere_radius,
                                                                                           camera_radius,
                                                                                           plotter)
                    vector_points_weight_pf[target].append(spherical_area_dc)
                    computed_area_by_hemisphere.append(spherical_area_dc)
                vector_points_pf[target] = np.row_stack((vector_points_pf[target],
                                                         np.concatenate((point_position,
                                                                         np.array([yaw, pitch, roll])))))
            if central_area_computed:
                for ca in computed_area_by_hemisphere:
                    vector_points_weight_pf[target].append(ca)
            central_area_computed = True

        points0 = vector_points_pf[target][:, :3]
        point_cloud0 = pv.PolyData(points0)
        plotter.add_mesh(point_cloud0)

        # cylinder.plot(show_edges=True)
        plotter.add_mesh(cylinder, show_edges=True)

        points = target_points_pf[target]
        point_cloud = pv.PolyData(points)
        plotter.add_mesh(point_cloud)

    plotter.show()
    return vector_points_pf


def get_points_route(vector_points_gpr: dict, route_gpr: dict):
    route_points = {}
    for target, data_s in route_gpr.items():
        route_points[target] = np.empty([0, 6])
        for data in data_s:
            point_start = vector_points_gpr[target][data[0]]
            point_end = vector_points_gpr[target][data[1]]
            route_points[target] = np.row_stack((route_points[target], point_end))
    return route_points


def plot_route(centroid_points_pf: dict, radius_pf: dict, target_points_pf: dict, vector_points_pr: dict):
    print('Starting showing data')
    # Create a plotter
    plotter = pv.Plotter()
    vector_points_pf = {}
    str_color = ['red', 'green', 'black']
    count_color = 0
    for target in centroid_points_pf.keys():
        cy_direction = np.array([0, 0, 1])
        n_resolution = 36
        cy_hight = height_proportion * np.max(target_points_pf[target][:, 2])
        r_mesh = radius_pf[target]
        h = np.cos(np.pi / n_resolution) * r_mesh
        l = np.sqrt(np.abs(4 * h ** 2 - 4 * r_mesh ** 2))

        # Find the radius of the spheres
        z_resolution = int(np.ceil(cy_hight / l))

        cylinder = pv.CylinderStructured(
            center=centroid_points_pf[target],
            direction=cy_direction,
            radius=r_mesh,
            height=1.0,
            theta_resolution=n_resolution,
            z_resolution=z_resolution,
        )

        points0 = vector_points_pr[target][:, :3]
        point_cloud0 = pv.PolyData(points0)
        plotter.add_mesh(point_cloud0, color=str_color[count_color])
        # arrow_direction = pv.Arrow(start=points0[0], direction=vector_points_pr[target][0, 3:])
        # plotter.add_mesh(arrow_direction, color=str_color[count_color])

        # cylinder.plot(show_edges=True)
        plotter.add_mesh(cylinder, show_edges=True)

        points = target_points_pf[target]
        point_cloud = pv.PolyData(points)
        plotter.add_mesh(point_cloud, color=str_color[count_color])
        count_color += 1

    plotter.show()


def quadcopter_control(sim, client, quad_target_handle, quad_base_handle, route_qc: dict):
    """
    This method is used to move the quadcopter in the CoppeliaSim scene to the position pos.
    :param quad_base_handle: The handle to get the quadcopter current position
    :param quad_target_handle:  The handle to the target of the quadcopter. This handle is used to position give the
    position that the quadcopter must be after control.
    :param orientation: The object orientation
    :param pos: The new position of the quadcopter is moved to.
    :return: A boolean indicating if the quadcopter reach the target position.
    """
    for target, position_orientation in route_qc.items():
        for i in range(position_orientation.shape[0]):
            cone_name = './' + target + f'/Cone'
            handle = sim.getObject(cone_name, {'index': i, 'noError': True})
            if handle < 0:
                break
            cone_pos = list(position_orientation[i, :3])
            sim.setObjectPosition(handle, cone_pos)
        for each_position in position_orientation:
            pos = list(each_position[:3])
            next_point_handle = sim.getObject('./new_target')
            sim.setObjectPosition(next_point_handle, pos)
            orientation = list(np.deg2rad(each_position[3:]))
            orientation_angles = [0.0, 0.0, orientation[0]]
            # sim.setObjectOrientation(quad_target_handle, [0.0, 0.0, orientation[0]], sim.handle_world)
            # while sim.getSimulationTime() < t_stab:
            #     print(sim.getObjectOrientation(quad_base_handle, sim.handle_world))
            #     client.step()
            #     continue
            # orientation_angles = sim.yawPitchRollToAlphaBetaGamma(orientation[0], orientation[2], orientation[1])
            # pos = sim.getObjectPosition(quad_base_handle, sim.handle_world)
            # camera_handle = sim.getObject('./O[0]/Cone[19]')
            # sim.setObjectOrientation(camera_handle, orientation_angles)
            # sim.setObjectPosition(camera_handle, pos)
            # client.step()
            # sim.setObjectOrientation(quad_target_handle, sim.handle_world, orientation_angles)
            total_time = sim.getSimulationTime() + settings['total simulation time']
            stabilized = False
            print('Change point')
            while sim.getSimulationTime() < total_time:
                diff_pos = np.subtract(pos, sim.getObjectPosition(quad_base_handle, sim.handle_world))
                norm_diff_pos = np.linalg.norm(diff_pos)
                if norm_diff_pos > 0.2:
                    delta_pos = 0.1 * diff_pos
                    new_pos = list(sim.getObjectPosition(quad_base_handle, sim.handle_world) + delta_pos)
                else:
                    new_pos = pos

                sim.setObjectPosition(quad_target_handle, new_pos)
                diff_ori = np.subtract(orientation_angles,
                                       sim.getObjectOrientation(quad_base_handle, sim.handle_world))
                norm_diff_ori = np.linalg.norm(diff_ori)

                if norm_diff_ori > 0.05:
                    delta_ori = 0.3 * diff_ori
                    new_ori = list(sim.getObjectOrientation(quad_base_handle, sim.handle_world) + delta_ori)
                else:
                    new_ori = orientation_angles
                sim.setObjectOrientation(quad_target_handle, new_ori)
                t_stab = sim.getSimulationTime() + settings['time to stabilize']
                while sim.getSimulationTime() < t_stab:
                    diff_pos = np.subtract(new_pos,
                                           sim.getObjectPosition(quad_base_handle, sim.handle_world))
                    diff_ori = np.subtract(new_ori,
                                           sim.getObjectOrientation(quad_base_handle, sim.handle_world))
                    norm_diff_pos = np.linalg.norm(diff_pos)
                    norm_diff_ori = np.linalg.norm(diff_ori)
                    if norm_diff_pos < 0.1 and norm_diff_ori < 0.05:
                        stabilized = True
                        break
                    client.step()
                diff_pos = np.subtract(pos, sim.getObjectPosition(quad_base_handle, sim.handle_world))
                diff_ori = np.subtract(orientation_angles, sim.getObjectOrientation(quad_base_handle, sim.handle_world))
                norm_diff_pos = np.linalg.norm(diff_pos)
                norm_diff_ori = np.linalg.norm(diff_ori)
                if norm_diff_pos < 0.1 and norm_diff_ori < 0.05:
                    stabilized = True
                    break
                client.step()
            if not stabilized:
                print('Time short')


def compute_edge_weight_matrix(S_cewm: dict) -> np.ndarray:
    print('Starting computing distance matrix')
    i = 0
    j = 0
    edge_weight_matrix = np.empty(0)
    for target_cewm, points_cewm in S_cewm.items():
        if i == 0 and j == 0:
            edge_weight_matrix = np.empty([points_cewm.shape[0] * 3, points_cewm.shape[0] * 3])
        for target_cewm_i, points_cewm_i in S_cewm.items():
            edge_weight_matrix[i, j] = np.linalg.norm(points_cewm[-1], target_cewm_i[-1])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # targets_border, targets_center, targets_points_of_view = create_sample_data()
    # points_of_view_contribution = camera_view_evaluation(targets_points_of_view)
    # S = subgroup_formation(targets_border, points_of_view_contribution, targets_points_of_view)
    # main_route = find_route(S)
    # save_points(main_route, targets_points_of_view)
    copp = CoppeliaInterface()
    positions, target_hull, centroid_points, radius = initializations(copp)
    targets_points_of_view = draw_cylinders_hemispheres(centroid_points, radius, positions)
    points_of_view_contribution = camera_view_evaluation(targets_points_of_view)
    S = subgroup_formation(target_hull, points_of_view_contribution, targets_points_of_view, positions)
    main_route = find_route(S)
    route_points = get_points_route(targets_points_of_view, main_route)
    plot_route(centroid_points, radius, positions, route_points)

    settings = parse_settings_file('config.yaml')
    copp.handles[settings['quadcopter name']] = copp.sim.getObject(settings['quadcopter name'])
    copp.handles[settings['quadcopter base']] = copp.sim.getObject(settings['quadcopter base'])
    quadcopter_control(copp.sim, copp.client, copp.handles[settings['quadcopter name']],
                       copp.handles[settings['quadcopter base']], route_points)
    # copp.sim.stopSimulation()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
