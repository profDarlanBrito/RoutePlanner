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
```