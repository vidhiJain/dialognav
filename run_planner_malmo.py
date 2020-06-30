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
import gym_minigrid.wrappers as wrappers
import pdb
import planner


OBS_TO_RECORD = ['xPos', 'zPos', 'yPos', 'yaw', 'pitch']



def write_to_csv(text,file_name):
    with open(file_name, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, lineterminator='\n')
        writer.writerow(text)

def getMissionXML(mission_file):
    with open(mission_file, 'r') as f:
        print("Loading mission from %s" % mission_file)
        mission_xml = f.read()

    return mission_xml

def make_dir(dir_name):
    if(not os.path.exists(dir_name)):
        os.mkdir(dir_name)

def get_actions(yaw):
    env = gym.make('MiniGrid-NumpyMapMinecraftUSAR-v')
    env = wrappers.VisdialWrapper(env)
    obs = env.reset()
    agent = planner.astar_planner(obs)
    goal = 8
    action_list = agent.Act(goal,obs,yaw=yaw,action_type="malmo")
    env.close()
    return action_list

def main():
    malmoutils.fix_print()
    agent_host = MalmoPython.AgentHost()
    malmoutils.parse_command_line(agent_host)
    recordingsDirectory = malmoutils.get_recordings_directory(agent_host)
    recordingsDirectory = "../human_trajectories"
    make_dir(recordingsDirectory)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG) # set to INFO if you want fewer messages

    video_width = 640
    video_height = 480

    mission_xml_path = "/home/akshay/Desktop/Research/visdial_malmo/malmo_037_py36_zip/missions/usar.xml"
    validate = True
    # my_mission = MalmoPython.MissionSpec(missionXML, validate)
    my_mission = MalmoPython.MissionSpec(getMissionXML(mission_xml_path), validate)
    agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
    # agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.KEEP_ALL_OBSERVATIONS)
    agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)

    if agent_host.receivedArgument("test"):
        num_reps = 1
    else:
        num_reps = 30000

    my_mission_record = MalmoPython.MissionRecordSpec()
    if recordingsDirectory:
        my_mission_record.recordRewards()
        my_mission_record.recordObservations()
        my_mission_record.recordCommands()
        # if agent_host.receivedArgument("record_video"): # my_mission_record.recordMP4(24,2000000)
        my_mission_record.recordMP4(24,2000000)



    for iRepeat in range(1):
        # my_mission_record.setDestination(os.path.join(curr_rec_dir, recording_name+".tgz"))
        max_retries = 3
        for retry in range(max_retries):
            try:
                agent_host.startMission( my_mission, my_mission_record )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    logger.error("Error starting mission: %s" % e)
                    exit(1)
                else:
                    time.sleep(2)

        logger.info('Mission %s', iRepeat)
        logger.info("Waiting for the mission to start")
        world_state = agent_host.getWorldState()
        
        print("world_state", world_state.has_mission_begun)
        while not world_state.has_mission_begun:
            world_state = agent_host.getWorldState()
            print(".", end="")
            time.sleep(0.1)
        print()
        yaw = 270
        action_list = get_actions(yaw)
        agent_host.sendCommand( "move 1" )
        while world_state.is_mission_running and len(action_list):
            world_state = agent_host.getWorldState()
            while world_state.number_of_video_frames_since_last_state < 1:
                logger.info("Waiting for frames...")
                time.sleep(0.05)
                world_state = agent_host.getWorldState()

            logger.info("Got frame!")
            

            world_state = agent_host.getWorldState()
            print("num_vid_frame ", world_state.number_of_video_frames_since_last_state)
            while world_state.number_of_video_frames_since_last_state > 1 and world_state.is_mission_running:
                video_frame = world_state.video_frames[-1]
                frame = get_image_frame(video_frame, height=480, width=640, depth=False)
                plt.imshow(frame)
                plt.show()
            action = action_list.pop()
            print(action)
            pdb.set_trace()
            agent_host.sendCommand(action)
            time.sleep(0.1)

def get_image_frame(video_frame, height, width, depth=False):                                                   
    pixels = video_frame.pixels
    img = Image.frombytes('RGB', (height, width), bytes(pixels))
    frame = np.array(img.getdata()).reshape(height, width,-1)
    return frame

if __name__=="__main__":
    main()

