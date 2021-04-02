#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

from gym_minigrid.minigrid import *

def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()
    # print(obs['image'].shape)

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)


def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
    coord = round(event.xdata / args.tile_size), round(event.ydata / args.tile_size)
    print(coord)


def step(action):
    obs, reward, done, info = env.step(action)
    # breakpoint()
    # print(obs.keys())
    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        reset()
    else:
        redraw(obs)
    # print('yaw', env.yaw)

def key_handler(event):
    print('pressed', event.key)

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
    default='MiniGrid-FourRoomsCluttered-v0'#'MiniGrid-MinimapForSparkySubmap-v0'  #'MiniGrid-GoToObject-10x10-N3-v0' #'MiniGrid-GoToObject-8x8-N2-v0' ## 'MiniGrid-MinimapForSparky-v0' #'MiniGrid-GoToDoor-8x8-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=100001
)
parser.add_argument(
    "-ms",
    "--map_set_number",
    type=int,
    help="map set for rendering",
    default=2
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

from pathlib import Path

RESOURCES_DIR = (Path(__file__).parent).resolve()

# env = gym_minigrid.envs.minimap.MinimapForSparky(
#         raw_map_path=Path(RESOURCES_DIR, 'gym_minigrid/envs/resources/map_set_'+str(args.map_set_number)+'.npy')
#     )
env = HumanFOVWrapper(env, agent_pos=(23, 14))
# env = InvertColorsWrapper(env)

# env = gym.wrappers.Monitor(env, "recording")
# env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True,force=True)
# if args.agent_view:
#     # env = RGBImgPartialObsWrapper(env)
#     env = ImgObsWrapper(env)

window = Window('gym_minigrid - ' + args.env + ' - Map set ' + str(args.map_set_number))
window.reg_key_handler(key_handler)
window.reg_click_event(onclick)
reset()

# Blocking event loop
window.show(block=True)