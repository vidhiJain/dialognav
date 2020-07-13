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


import os
import sys
import pdb
import numpy as np
import imageio
import os
import time
from utils import *

OBS_TO_RECORD = ['xPos', 'zPos', 'yPos', 'yaw', 'pitch']


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
