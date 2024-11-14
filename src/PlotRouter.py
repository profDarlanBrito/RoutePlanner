import ast
import csv
import pickle

import numpy as np
import pyvista as pv


def get_cops_data(cops_path: str):
    with open(cops_path, "r") as file:
        for line in file:
            line_split = line.split(": ")
            key = line_split[0]
            content = line_split[1]

            if key == "DIMENSION":
                dimension = int(content.strip())

            if key == "CLUSTERS":
                clusters = int(content.strip())

            if key == "SUBGROUPS":
                subgroups = int(content.strip())

            if key == "NODE_COORD_SECTION":

                key_points = {}
                for _ in range(dimension):
                    line = file.readline()
                    id_vertex, x, y, z = line.split(" ")
                    key_points[int(id_vertex)] = np.array([float(x), float(y), float(z)])

            if key == "GTSP_SUBGROUP_SECTION":
                key_subgroup = {}
                for _ in range(subgroups):
                    line = file.readline()
                    cluster_id, cluster_profit, *id_vertex_list = line.split(" ")
                    id_vertex_list = list(map(int, id_vertex_list))
                    key_subgroup[int(cluster_id)] = (float(cluster_profit), id_vertex_list)

            if key == "GTSP_CLUSTER_SECTION":
                key_cluster = {}
                for _ in range(clusters):
                    line = file.readline()
                    set_id, *id_cluster_list = line.split(" ")
                    id_cluster_list = list(map(int, id_cluster_list[:-1]))
                    key_cluster[int(set_id)] = id_cluster_list

    return {"points": key_points, "subgroups": key_subgroup, "clusters": key_cluster}


def get_result_cops_route(result_cops_path: str, key_points: dict):
    with open(result_cops_path, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=";")
        _ = next(reader)  # header
        line = next(reader)  # first line content
        line = line[7].replace("  ", ", ")
        cops_route = ast.literal_eval(line)

    route = []
    route.append(key_points[cops_route[0][0]])

    for _, key_id in cops_route:
        route.append(key_points[key_id])

    return route


# Load data from files
exp = 2
sub_exp = 0
workspace = f"save_files/0"
convex_hull_file_path = f"{workspace}/variables/convex_hull_{sub_exp}.var"
cops_file_path = f"{workspace}/datasets/3dreconstructionPathCOPS{sub_exp}.cops"
results_cops_file_path = f"{workspace}/datasets/results/3dreconstructionPathCOPS{sub_exp}.csv"

with open(convex_hull_file_path, "rb") as file:
    S = pickle.load(file)
    targets_points_of_view = pickle.load(file)
    centroid_points = pickle.load(file)
    radius = pickle.load(file)

keys = list(centroid_points.keys())
centroids = [centroid_points[k] for k in keys]
radius = [radius[k] for k in keys]

cylinders = [(radius, 2 * point[2], *point) for radius, point in zip(radius, centroids)]

cops_data = get_cops_data(cops_file_path)
route_points = get_result_cops_route(results_cops_file_path, cops_data["points"])

# Create a plotter
plotter = pv.Plotter()

# Add the cylinders
for r, h, x, y, z in cylinders:
    cylinder = pv.CylinderStructured(center=(x, y, z), direction=(0, 0, 1), radius=r, height=h)
    plotter.add_mesh(cylinder, show_edges=True)

# Add the route line
for i, r in enumerate(route_points[1:]):
    line = pv.Line(route_points[i], route_points[i + 1])
    plotter.add_mesh(line, color="blue", line_width=3)

for point in route_points:
    sphere = pv.Sphere(radius=0.05, center=point)
    plotter.add_mesh(sphere)

# Define primary colors for clusters
primary_colors = [
    [255, 0, 0],  # Red
    [0, 255, 0],  # Green
    [0, 0, 255],  # Blue
]

# for cluster_id, subgroup_ids in cops_data['clusters'].items():
#     color = primary_colors[cluster_id % len(primary_colors)]  # Cycle through primary colors
#     for subgroup_id in subgroup_ids:
#         _, point_ids = cops_data['subgroups'][subgroup_id]  # Get the points in the subgroup
#         for point_id in point_ids:
#             point = cops_data['points'][point_id]
#             sphere = pv.Sphere(radius=0.05, center=point)
#             # plotter.add_mesh(sphere, color=color)
#             plotter.add_mesh(sphere)

# for cluster_id, subgroup_ids in cops_data['clusters'].items():
#     color = primary_colors[cluster_id % len(primary_colors)]  # Cycle through primary colors

#     for point_id in cops_data['subgroups'][2][1]:
#         point = cops_data['points'][point_id]
#         sphere = pv.Sphere(radius=0.1, center=point)
#         plotter.add_mesh(sphere, color=color)

# Show grid and plot
plotter.show_grid()
plotter.show()
