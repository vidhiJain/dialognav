import random
import numpy as np
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX
import gym_minigrid.wrappers as wrappers
import pdb
import planner

#making the environment.
# env = gym.make('MiniGrid-MultiRoom-N6-v0')
# env = gym.make('MiniGrid-NumpyMap-v0')
# env = gym.make('MiniGrid-NumpyMapMinecraftUSAR-v0')
env = gym.make(MiniGrid-MinimapForSparky-v0)
# env = RGBImgPartialObsWrapper(env)
env = wrappers.FullyObsWrapper(env)
obs = env.reset()
env.render()
agent = planner.astar_planner(obs)
done = False

pdb.set_trace()
while (not done):
    action = agent.Act(obs=obs, goal=8, action_type="minigrid")
    # print('action', action)
    obs, reward, done, info = env.step(action)
    env.render()



