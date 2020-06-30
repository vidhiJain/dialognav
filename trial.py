import random
import numpy as np
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX
import gym_minigrid.wrappers as wrappers
from gym_minigrid.window import Window
import planner
import pdb

#making the environment.
# env = gym.make('MiniGrid-MultiRoom-N6-v0')
# env = gym.make('MiniGrid-NumpyMap-v0')
env = gym.make('MiniGrid-MinimapForSparky-v0')
# env = RGBImgPartialObsWrapper(env)

# env = wrappers.FullyObsWrapper(env)
env = wrappers.VisdialWrapper(env)
window = Window('gym_minigrid - ' + 'MiniGrid-MinimapForSparky-v0')
obs = env.reset()
# env.render()
img = env.render()
window.show_img(img=img)
agent = planner.astar_planner(obs)
done = False

while (True):
    action = agent.Act(obs=obs, goal=8, action_type="minigrid")
    #we have reached if the action is -1.
    if action == -1:
    	print("reached goal")
    	break
    obs, reward, done, info = env.step(action)
    # env.render()
    img = env.render()
    window.show_img(img=img)
