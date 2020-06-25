from pathlib import Path
import sys

# Add `gym_minigrid` parent directory to system path
sys.path.append(str(Path(__file__).parent.parent.resolve().absolute()))

# Import the envs module so that envs register themselves
import gym_minigrid.envs

# Import wrappers so it's accessible when installing with pip
import gym_minigrid.wrappers
