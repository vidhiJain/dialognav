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


def getMissionXML(mission_file):
    with open(mission_file, 'r') as f:
        print("Loading mission from %s" % mission_file)
        mission_xml = f.read()

    return mission_xml

def main():
    malmoutils.fix_print()
    agent_host = MalmoPython.AgentHost()
    malmoutils.parse_command_line(agent_host)
    recordingsDirectory = malmoutils.get_recordings_directory(agent_host)
    recordingsDirectory = "../human_trajectories"
    if(not os.path.exists(recordingsDirectory)):
        os.mkdir(recordingsDirectory)
    logging.basicConfig(level=logging.INFO)
    # pdb.set_trace()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG) # set to INFO if you want fewer messages

    video_width = 640
    video_height = 480
    sys.argv

    mission_xml_path = "../custom_xmls/usar.xml"
    validate = True
    # my_mission = MalmoPython.MissionSpec(missionXML, validate)
    my_mission = MalmoPython.MissionSpec(getMissionXML(mission_xml_path), validate)
    agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
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
    recording_name = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    for iRepeat in range(1):
        my_mission_record.setDestination(os.path.join(recordingsDirectory, recording_name+".tgz"))
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
        while not world_state.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
        print()

        img_counter = 0

        while world_state.is_mission_running:
            world_state = agent_host.getWorldState()
            while world_state.number_of_video_frames_since_last_state < 1 and world_state.is_mission_running:
                logger.info("Waiting for frames...")
                time.sleep(0.05)
                world_state = agent_host.getWorldState()

            logger.info("Got frame!")
            
            if world_state.is_mission_running:
                timestamp = world_state.observations[-1].timestamp
                msg = world_state.observations[-1].text
                print(timestamp)
                print(msg)
                frame = world_state.video_frames[-1]
                img = Image.frombytes('RGB', (640,480), bytes(frame.pixels))
                # imageio.imsave("./tmp_imgs/{}.png".format(img_counter), img)
                img_counter += 1
        logger.info("Mission has stopped.")
        time.sleep(1) # let the Mod recover

if __name__=="__main__":
    main()

