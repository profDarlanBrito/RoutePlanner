## Commands Singularity 4
#### sandbox
```
singularity build --sandbox route_planner/ docker://index.docker.io/library/ubuntu:20.04
```
#### edit
```
singularity shell --nv --writable --fakeroot --no-home route_planner/
```
#### edit env inside singularity container  
```
/.singularity.d/env/91-environment.sh
```
#### build
```
nohup singularity build --fakeroot --fix-perms route_planner.sif route_planner/ &
```
#### run
```
nohup singularity exec --nv --no-home --fakeroot --bind [bind_path]:/mnt route_planner.sif /root/RoutePlanner/RouteOptimization.sh [commands] &
```

#### run from folder
```
nohup singularity exec --nv --writable --no-home --fakeroot --bind [bind_path]:/mnt route_planner /root/RoutePlanner/RouteOptimization.sh [commands] &
```