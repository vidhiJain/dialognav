import numpy as np
import csv
import os
import pdb
import sys
import argparse
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX
from gym_minigrid.window import Window
import gym_minigrid.wrappers as wrappers
import planner
import malmoutils
from builtins import range
from past.utils import old_div
import MalmoPython
import logging
import struct
import socket

## generic functions
def write_to_csv(text,file_name):
    with open(file_name, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, lineterminator='\n')
        writer.writerow(text)


def make_dir(dir_name):
    if(not os.path.exists(dir_name)):
        os.mkdir(dir_name)

## minigrid related functions
def redraw(env, img, window):
    img = env.render('rgb_array')
    window.show_img(img)

def start_minigrid_env():
    # starts a minigrid env, and returns the env object, the first obs, and the window object

    env = gym.make('MiniGrid-MinimapForSparky-v0')
    env = wrappers.VisdialWrapper(env)
    obs = env.reset()
    window = Window('gym_minigrid - ' + 'MiniGrid-MinimapForSparky-v0')
    redraw(env, obs['image'], window)
    return env, obs, window

def take_minigrid_action(env, action, window):
    # takes the given action in the env, and updates the rendered view in the window
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

## malmo related functions
def get_image_frame(video_frame, height, width, depth=False):                                                   
    # returns the current observed view of malmo, as a np array
    pixels = video_frame.pixels
    img = Image.frombytes('RGB', (height, width), bytes(pixels))
    frame = np.array(img.getdata()).reshape(height, width,-1)
    return frame

def getMissionXML(mission_file):
    with open(mission_file, 'r') as f:
        print("Loading mission from %s" % mission_file)
        mission_xml = f.read()
    return mission_xml

def start_planner_agent(obs):
    # obs: the first obs of the minigrid env after reset
    #  returns a planner agent for the given obs
    agent = planner.astar_planner(obs)
    return agent

def start_malmo_env():
    ## starts a malmo env, and returns the agent host for the env
    malmoutils.fix_print()
    
    # create agent host
    agent_host = MalmoPython.AgentHost()
    malmoutils.parse_command_line(agent_host)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG) # set to INFO if you want fewer messages

    video_width = 640
    video_height = 480
    
    mission_xml_path = "mission_xmls/usar.xml"
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
