import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import pipeline
import pdb
import gym_minigrid
from gym_minigrid.wrappers import VisdialWrapper, FullyObsWrapper
from gym_minigrid.window import Window
import gym
import planner
import argparse
import time
from utils import *
import nav_instruction_parser as instruction
# breakpoint()
parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-MinimapForSparky-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=8
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)

def redraw(img):
    # if not args.agent_view:
    img = env.render('rgb_array')
    window.show_img(img)
def reset():

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs['image_fov'])

def step(action):
    obs, reward, done, info = env.step(action)
    if done:
        reset()
        # env.put_obj(Goal('blue'), env.agent_pos[0], env.agent_pos[1])
    else:
        redraw(obs['image_fov'])
args = parser.parse_args()

#environment
env = gym.make('MiniGrid-MinimapForSparky-v0')
env = VisdialWrapper(env, tile_size=8, agent_pos=(24,25))
window = Window('gym_minigrid - ' + 'minigrid-minimapforsparky-v0')
# window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=False)

malmo_agent = start_malmo_env()

# malmo_start_position = (-2133.5,52.0,175.5)
# malmo_start_orientation = (0.0, 90, 0.0)
origin_coordinates = {'x': -2133.5 - 25, 'z': 175.5 - 24}

#language processing
instructionProcessor = instruction.InstructionProcessor()

#a* based agent.
obs = env.reset()
agent = planner.AstarPlanner(obs)

world_state = malmo_agent.getWorldState()

while not world_state.has_mission_begun:
    world_state = malmo_agent.getWorldState()
    print(".", end="")
    time.sleep(0.1)
try:
    while True:
        malmo_action = None
        instruction = input("Enter the instruction: ")
        goalTuple, previous, immediate_actions = instructionProcessor.defineGoal(instruction)
        print('goalTuple', goalTuple, 'previous', previous, 'immediate_actions', immediate_actions)
        while malmo_action!="done":
            # If malmo env
            if malmo_agent is not None:
                if not world_state.is_mission_running:
                    break
                world_state = malmo_agent.getWorldState()
                while world_state.number_of_video_frames_since_last_state < 1:
                    time.sleep(0.5)
                    world_state = malmo_agent.getWorldState()
                time.sleep(1)
                world_state = malmo_agent.getWorldState()
            
            ## get agent_pose
            malmo_agent_pose = get_agent_pose(world_state)
            print('malmo_agent_pose', malmo_agent_pose)

            # TODO: apply sanity check that the minimap and malmo are in sync or not
            if env.agent_pos[0] != malmo_agent_pose['position'][2]-origin_coordinates['z']:
                if env.agent_pos[1] != malmo_agent_pose['position'][0]-origin_coordinates['x']:
                    breakpoint()


            path = "./temp"

            action, malmo_action = agent.Act(goal=goalTuple, obs=obs, 
                                                action_type="malmo",previous=previous)
            # If malmo env 
            if malmo_agent is not None:
                malmo_agent.sendCommand(malmo_action)
            
            obs = take_minigrid_action(env, action, window)
            time.sleep(0.5)

except KeyboardInterrupt:
    pass