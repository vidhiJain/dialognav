#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import VisdialWrapper, FullyObsWrapper
from gym_minigrid.window import Window

import planner
import pdb
from dailog import process_dialog

def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array')
    window.show_img(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()
    # print(obs['image'].shape)

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs['image_fov'])

def step(action):
    obs, reward, done, info = env.step(action)
    # print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        # print('done!')
        reset()
        # env.put_obj(Goal('blue'), env.agent_pos[0], env.agent_pos[1])
    else:
        redraw(obs['image_fov'])

def key_handler(event):
    # print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-MultiRoom-N6-v0'
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

args = parser.parse_args()

env = gym.make(args.env)
env = VisdialWrapper(env, args.tile_size)
# env = FullyObsWrapper(env)

# TO RECORD
# env = gym.wrappers.Monitor(env, "recording")
#env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True,force=True)

if args.agent_view:
    #env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=False)

# ------------------------------------------------
def start_dialog(env, window=None):
    obs = env.reset()
    #this observation is just passed to intialise the parameters of the planner.
    agent = planner.astar_planner(obs)

    while True:
        ip = input(">> ").lower().strip()
        if ip == "quit":
            break
        response = process_dialog(ip, agent, env, window)
        print(response)
    print("Done")

start_dialog(env, window)
