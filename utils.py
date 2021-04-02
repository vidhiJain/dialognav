import sys
import numpy as np
import csv
import os
import pdb
import argparse
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX
from gym_minigrid.window import Window
import gym_minigrid.wrappers as wrappers
import planner
from builtins import range
from past.utils import old_div
import logging
import struct
import socket
import time
import json
# from numpyencoder import NumpyEncoder
import pdb
import malmoutils
import MalmoPython
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

def get_agent_pose(world_state):
    # world_state = agent_host.getWorldState()
    # while world_state.number_of_video_frames_since_last_state < 1:
    #     time.sleep(0.5)
    #     world_state = agent_host.getWorldState()
    # # breakpoint()
    # while world_state is None:
        # time.sleep(0.5)
    msg = world_state.observations[-1].text
    observations = json.loads(msg)

    XPos, YPos, ZPos = observations['XPos'], observations['YPos'], observations['ZPos']
    Pitch, Yaw, Roll = observations['Pitch'], observations['Yaw'], 0.0
    pose = {'position': (XPos, YPos, ZPos),
            'orientation': (Pitch, Yaw, Roll),
            }

    return pose

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

def start_malmo_env(record_obs=True, record_video=True, mission_xml_path="mission_xmls/usar.xml"):
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
    validate = True
    my_mission = MalmoPython.MissionSpec(getMissionXML(mission_xml_path), validate)

    if(record_obs):
        agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
    if(record_video):
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

class fov_generator():

    def __init__(self, position=(-2133.5,52.5,175.5),
            orientation=(0.0,90,0.0),
            window_size=(640,480),
            json_path="/home/akshay/Desktop/Research/dana_fov_stuff/pygl_fov/data/parsed_world_temp.json",
            ):

        import pygl_fov
        import pygl_fov.visualize
        import pygl_fov.summaries
        self.position = position
        self.orientation = orientation
        self.feeder = pygl_fov.BlockListFeeder.loadFromJson(json_path)
        self.perspective = pygl_fov.Perspective(position=position,
                orientation=orientation,
                window_size=window_size)
        self.perspective.playerState.fovy = 110
        self.fov = pygl_fov.FOV(self.perspective, self.feeder)
        self.fov.prepareVBO()

        pltWindow = pygl_fov.visualize.MatplotWindow()
        self.semanticVis = pygl_fov.visualize.SemanticMapVisualizer(window_size)
        pltWindow.add(self.semanticVis,111)
        self.summary = pygl_fov.summaries.BlockListSummary(self.fov)
        
        self.blocks_info_dict = {}

    def update_summary(self, agent_position, agent_orientation):

        self.perspective.set_pose(agent_position, agent_orientation)
        pixelMap = self.fov.calculatePixelToBlockIdMap()
        semanticMap = np.zeros(pixelMap.shape)
        for blockID in np.unique(pixelMap):
            if blockID != -1 and blockID<len(self.feeder):
                semanticMap[pixelMap==blockID] = self.feeder[blockID].block_type


        self.semanticVis.values = semanticMap
        self.semanticVis.update(pause_plt=False)


    def get_block_data(self):
        blocks_data_list = []
        for block in self.summary():
            block_dict = block.__dict__
            block_dict['block_type'] = block_dict['block_type'].name
            blocks_data_list.append(block_dict)

        blocks_data = {"blocks": blocks_data_list}
        return blocks_data_list

    def save_fov_img(self, save_path=None):
        if(save_path):
            self.semanticVis.save_img(save_path)
    
    def collect_block_info(self, timestamp):
        block_data = self.get_block_data()
        self.blocks_info_dict[timestamp] = block_data

    # def save_data(self, save_path=None):
    #     if(save_path):
    #         with open(save_path, 'w') as jsonFile:
    #             json.dump(self.blocks_info_dict, jsonFile,cls=NumpyEncoder, indent=2)
