<!-- README.md -->
# VisDial for Minecraft USAR 

## Setup

1. Ensure Malmo Mod for Minecraft is set 
2. Run Minecraft 
3. Collected trajectories saved in `.tgz` file in folder `human_trajectories`.
4. Install gym_minigrid with dependencies with `setup.py`. 
```
cd code 
pip install -e .
```



## Run
Change the xmlfile name in `FileWorldGenerator`

```
cd code 
python3 minecraft_to_minigrid_debug.py --xmlfile <path_to_xml>
```

Play the game to record the observed map and directional frontiers at each timestep.
TODO: Record over some time interval like 5 steps?


## Data structure 
numpy array saved with `map`  which is env_size x envsize, `agent_dir` which is continuous yaw, `subset_frontiers_mask` which is a boolean list of frontier, where the frontiers are calculated from top left to bottom right. Refer `get_absolute_frontier_coordinates` in `minecraft_to_minigrid_debug.py`


We can also save the 3D tensor with channels as `map` and `subset_frontiers`.


## Input format
3D tensor of shape env_size, env_size, num_channels
where the channels contain the following :
index 0 : trajectory map of the agent in absolute coordinates
index 1 : subset of frontiers selected 

Note: currently frontiers are selected based on cardinal directional input. To be extended to natural language input.

