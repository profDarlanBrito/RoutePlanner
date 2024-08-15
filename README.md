# RoutePlanner

## Contents

- [Using a Virtual Environment](#using-a-virtual-environment)
- [Dependencies](#dependencies)  
- [Usage](#usage)


## Using a Virtual Environment

### Create an isolated virtual environment

`virtualenv <folder>` A folder will be created with the specified name, containing the subfolders bin, include, lib, and local.


For example, to create a virtual environment in the folder .venv

```
python -m venv .venv
```

Activate an isolated virtual environment
```
source .venv/bin/activate
```

To know if you are inside the virtual environment, the prompt will change to something in the format (.venv) ...

Deactivate an isolated virtual environment
```
deactivate
```


## Dependencies

- Python 3.11.0
- Another dependencies in requirements.txt
```
pip install -r requirements.txt
```

## Usage 
Optional available arguments:
```txt
convex_hull (Generates an optimized route around objects in the scene using COPS)
view_point (Captures images of objects in the scene using CoppeliaSim)
point_cloud (Performs reconstruction from the images)
poisson_check (Check if the Poisson mesh was generated, if not, create it using `fused.ply` file)
```
To execute, use the RouteOptimization.py script, for example:
```
python RouteOptimization.py <arguments>
```
If no arguments are provided, all three processes will be executed sequentially:

```
# Run all processes sequentially
python RouteOptimization.py

# Run individual processes:
python RouteOptimization.py convex_hull
python RouteOptimization.py view_point 
python RouteOptimization.py point_cloud
python RouteOptimization.py poisson_check
python RouteOptimization.py mesh_analysis
```

## Commands Singularity 4
#### sandbox
```
singularity build --sandbox route_planner/ docker://index.docker.io/library/ubuntu:20.04
```
#### edit
```
singularity shell --nv --writable --fakeroot --no-home --bind /homeLocal/othiago/local_mnt/:/mnt route_planner/
```
#### edit env
```
/.singularity.d/env/91-environment.sh
```
#### build
```
singularity build --fakeroot --fix-perms route_planner.sif route_planner/
```
#### run
```
nohup singularity exec --nv --no-home --fakeroot --bind [bind_path]:/mnt route_planner.sif /root/RoutePlanner/RouteOptimization.sh [commands] &
```