from __future__ import print_function
from __future__ import division # ------------------------------------------------------------------------------------------------ # Copyright (c) 2016 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------


from builtins import range
from past.utils import old_div
import MalmoPython
import random
import time
import logging
import struct
import socket
import os
import sys
import malmoutils
import pdb
import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageTk
import os
import argparse
import sys 
from datetime import datetime
import csv
import json
import matplotlib.pyplot as plt
import torch 
import gym
import random
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX
from gym_minigrid.window import Window
import gym_minigrid.wrappers as wrappers
import pdb
import planner
from utils import *

OBS_TO_RECORD = ['xPos', 'zPos', 'yPos', 'yaw', 'pitch']

def redraw(env, img, window):
    img = env.render('rgb_array')
    window.show_img(img)

def start_minigrid_env():
    env = gym.make('MiniGrid-MinimapForSparky-v0')
    env = wrappers.VisdialWrapper(env)
    obs = env.reset()
    window = Window('gym_minigrid - ' + 'MiniGrid-MinimapForSparky-v0')
    redraw(env, obs['image'], window)
    return env, obs, window

def start_planner_agent(obs):
    agent = planner.astar_planner(obs)
    return agent

# def get_action(env, obs, agent, goal_id, window):
    # goal = goal_id
    # minigrid_action, malmo_action= agent.Act(goal,obs,action_type="malmo") # action_list is in reverse order
    
    # return minigrid_action, malmo_action

def take_minigrid_action(env, action, window):
    if(action==-1):
        obs = env.gen_obs()
    else:
        obs, _, _, _ = env.step(action) # obs: minigrid observation
        redraw(env, obs['image'], window)
        if(action==0):
            caption = "turn 90 anticlockwise"
        elif(action==1):
            caption = "turn 90 clockwise"
        elif(action==2):
            caption = "move forward"
        elif(action==5):
            caption = "toggle"
        else:
            caption = "task finished"

        window.set_caption(caption)

    return obs

def start_malmo_env():

    malmoutils.fix_print()
    
    # create agent host
    agent_host = MalmoPython.AgentHost()
    malmoutils.parse_command_line(agent_host)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG) # set to INFO if you want fewer messages

    video_width = 640
    video_height = 480
    
    # start minigrid env

    # mission_xml_path = "usar.xml"
    mission_xml_path = "/home/akshay/Desktop/Research/visdial_malmo/malmo_037_py36_zip/missions/usar.xml"
    validate = True
    my_mission = MalmoPython.MissionSpec(getMissionXML(mission_xml_path), validate)
    agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
    agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)

    if agent_host.receivedArgument("test"):
        num_reps = 1
    else:
        num_reps = 30000

    my_mission_record = MalmoPython.MissionRecordSpec()
    
    # starting mission
    max_retries = 3
    for retry in range(max_retries):
        try:
            agent_host.startMission(my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                logger.error("Error starting mission: %s" % e)
                exit(1)
            else:
                time.sleep(2)
    return agent_host 

def run_malmo(agent_host, env, minigrid_obs, planner_agent, minigrid_window, goal_id):
    terminate_mission = False
    # for iRepeat in range(1):
    while not terminate_mission:
        # logger.info("Waiting for the mission to start")
        world_state = agent_host.getWorldState()
        
        while not world_state.has_mission_begun:
            world_state = agent_host.getWorldState()
            print(".", end="")
            time.sleep(0.1)
        print()
        # action_list = get_actions(env, planner_agent)
        minigrid_action, action = planner_agent.Act(goal_id, minigrid_obs, action_type="malmo")
        minigrid_obs = take_minigrid_action(env, minigrid_action, minigrid_window)
        while world_state.is_mission_running and action!="done":
            world_state = agent_host.getWorldState()
            while world_state.number_of_video_frames_since_last_state < 1:
                time.sleep(0.05)
                world_state = agent_host.getWorldState()
            
            world_state = agent_host.getWorldState()
            print("action: {}".format(action))
            agent_host.sendCommand(action)
            # action, minigrid_obs = get_action(env, minigrid_obs, planner_agent, goal_id, minigrid_window)
            minigrid_action, action = planner_agent.Act(goal_id, minigrid_obs, action_type="malmo")
            minigrid_obs = take_minigrid_action(env, minigrid_action, minigrid_window)
            time.sleep(0.5)
        
        user_input = input("** awaiting new command: ").lower()
        if(user_input=="end"):
            terminate_mission = True
        else:
            goal_id = int(input("goal id: "))

def main():

    env, minigrid_obs, minigrid_window = start_minigrid_env()
    planner_agent = start_planner_agent(minigrid_obs)
    agent_host = start_malmo_env()
    goal_id = 8
    run_malmo(agent_host, env, minigrid_obs, planner_agent, minigrid_window, goal_id)

if __name__=="__main__":
    main()
