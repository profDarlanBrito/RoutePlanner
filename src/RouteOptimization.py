import multiprocessing
import os
import pickle
import sys

import Config
from Reconstruction.ConvexHull import convex_hull, load_variables
from CoppeliaInterface import CoppeliaInterface
from MeshAnalysis import mesh_analysis
from Reconstruction.PointCloud import generate_poisson_mesh, point_cloud
from Reconstruction.ViewPoint import view_point

settings = Config.Settings.get()


def update_current_experiment(value_stage: int) -> None:
    with open(os.path.join(settings['save path'], '.progress'), 'wb') as file:
        pickle.dump(value_stage, file)


def execute_experiment() -> None:
    # Create the directory
    os.makedirs(os.path.join(settings['save path'], 'variables'), exist_ok=True)

    with open(os.path.join(settings['save path'], '.progress'), 'rb') as f:
        last_expe = pickle.load(f)

    try:
        if len(sys.argv) < 2:

            for experiment in range(settings['number of trials']):
                if experiment < last_expe:
                    continue

                proc = multiprocessing.Process(target=convex_hull, args=(experiment,))
                proc.start()
                proc.join()

                copp = CoppeliaInterface(settings)
                view_point(copp, experiment)

                copp.sim.stopSimulation()
                del copp

                point_cloud(experiment)

                generate_poisson_mesh(experiment)

                mesh_analysis()
                update_current_experiment(int(experiment + 1))

            os.remove(os.path.join(settings['save path'], '.progress'))
            return

        if sys.argv[1] == 'convex_hull':
            for experiment in range(settings['number of trials']):
                if experiment < last_expe:
                    continue

                proc = multiprocessing.Process(target=convex_hull, args=(experiment,))
                proc.start()
                proc.join()
                update_current_experiment(int(experiment + 1))

            os.remove(os.path.join(settings['save path'], '.progress'))
            return

        if sys.argv[1] == 'view_point':
            copp = CoppeliaInterface(settings)
            for experiment in range(settings['number of trials']):
                if experiment < last_expe:
                    continue

                view_point(copp, experiment)
                update_current_experiment(int(experiment + 1))

            os.remove(os.path.join(settings['save path'], '.progress'))
            copp.sim.stopSimulation()
            return

        if sys.argv[1] == 'point_cloud':
            for experiment in range(settings['number of trials']):
                if experiment < last_expe:
                    continue

                point_cloud(experiment)
                update_current_experiment(int(experiment + 1))

            os.remove(os.path.join(settings['save path'], '.progress'))
            return

        if sys.argv[1] == 'poisson_check':
            generate_poisson_mesh()
            return

        if sys.argv[1] == 'mesh_analysis':
            mesh_analysis()
            return

    except RuntimeError as e:
        print("An error occurred:", e)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    load_variables()

    save_path = settings['save path']

    os.makedirs(save_path, exist_ok=True)

    # check if file not exits
    progress_file = os.path.join(settings['save path'], '.progress')
    if not os.path.isfile(progress_file):
        with open(progress_file, 'wb') as file:
            pickle.dump(0, file)

    execute_experiment()