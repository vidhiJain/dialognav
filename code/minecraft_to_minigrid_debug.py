import MalmoPython
import os
import sys
import time
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import utils.searchutils
import utils.search

import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

from gym_minigrid.minigrid import *



print('ACTIVE WARNING TO MAKE SURE:  same init position for agent in both usar and minigrid env')

map_grid_custom_colors = { 
        255: [200,200,200],
        # 0.5: [150,150,150],  
        0: [0,0,0],
        1: [0,0,255],
        2: [63,165,76 ], 
        3: [51, 153,245],
        4: [98, 67, 67],
        5: [255, 196, 75], 
        6: [140, 40, 40],
        7: [255, 0, 0],
        8: [0, 0, 255],
        9: [255,255, 255],
        10: [70, 70, 70]
    }

visible_grid_custom_colors = {
        0: [70,70,70],   # [200,200,200],
        # 0.5: [150,150,150],  
        1: [255, 255, 255],
        8: [0, 0, 255]
    }








def getMissionXML(mission_file):
    with open(mission_file, 'r') as f:
        print("Loading mission from %s" % mission_file)
        mission_xml = f.read()
    return mission_xml


def plot_with_custom_colors(image, name='debug_plot'):
    if image is None:
        return 

    plt.clf()
    plt.xlabel('x axis')
    plt.ylabel('z axis')
    
    plt.subplot(131)
    # image1 = np.array([[d[val] for val in row] for row in a[0]], dtype='B')
    plt.imshow(image[0])
    plt.title('below') # for fires')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(132)
    # image2 = np.array([[d[val] for val in row] for row in a[1]], dtype='B')
    plt.imshow(image[1])
    plt.title('same level') # for stone_buttons')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(133)
    # image3 = np.array([[d[val] for val in row] for row in a[2]], dtype='B')
    plt.imshow(image[2])
    plt.title('above') # for levers
    plt.xticks([])
    plt.yticks([])

    # plt.ion()
    plt.show()
    plt.pause(0.1)
    plt.savefig(name)
    # image = np.array([[d[val] for val in row] for row in a], dtype='B')
    # plt.imshow(image)

    return 


def plot2D(a, custom_colors, name):
    d = custom_colors
    plt.clf()    
    plt.xlabel('x axis')
    plt.ylabel('z axis')
    image = np.array([[d[val] for val in row] for row in a], dtype='B')
    plt.imshow(image)
    plt.title('2D map') # for fires')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    # plt.pause(5)
    plt.savefig(name)
    return 


class PlanningAgent():
    def __init__(self, xmlfile):

        self.agent_host = MalmoPython.AgentHost()
        self.my_mission = MalmoPython.MissionSpec(getMissionXML(xmlfile), True)
        self.my_mission_record = MalmoPython.MissionRecordSpec()

        self.objects_of_interest = ['stone_button', 'wooden_door', 'lever']

        # 4 represents anything in the env that is walkable (excluding wool)
        self.object_to_index = {'air': 9, 'player':8, 'wooden_door':2, 'wool': 3, 
            'stained_hardened_clay': 4, 'clay': 4, 'iron_block': 4, 'quartz_block': 4,
             'fire':5, 'lever':6, 'stone_button': 7, 'gravel': 10, 'redstone_wire': 4}

        self.index_to_object = { 255: 'unknown', 9: 'frontier',  8: 'player',  2: 'wooden_door',  3: 'wool', 
             4: 'wall',
             5: 'fire',  6: 'lever',  7: 'stone_button',  10: 'gravel'}

        self.non_opaque_objects = [9, 8, 1, 2, 5, 6, 7] #state of the door to be recorded
        self.passable_objects = ['air', 'wooden_door'] #, 'lever', 'gravel']
        self.passable_objects_with_cost = {'air': 1,  'lever': 1, 'wooden_door': 2, 'gravel': 5}
        self.floor_objects_types = ['redstone_wire', 'wool', 'iron_block', 'quartz_block']      
        self.envsize = 50
        # Env specific variables; (modify them wrt xmlfile)
        # self.sight= {'x': (-3, 3), 'z': (-3, 3), 'y':(-1, 1)}
        self.sight= {'x': (-21, 21), 'z': (-21, 21), 'y':(-1, 1)}
        self.angle = 50
        self.range_x = abs(self.sight['x'][1] - self.sight['x'][0]) + 1
        self.range_y = abs(self.sight['y'][1] - self.sight['y'][0]) + 1
        self.range_z = abs(self.sight['z'][1] - self.sight['z'][0]) + 1
        self.my_mission.observeGrid(self.sight['x'][0], self.sight['y'][0], self.sight['z'][0], 
            self.sight['x'][1], self.sight['y'][1], self.sight['z'][1], 'relative_view')
        self.scanning_range = 15
        
        # Goal specific variables
        self.num_victims_seen = 0
        self.num_doors_seen = 0
        self.total_victims = 3
        self.total_doors = 3
        self.victims_visited = np.zeros((self.envsize, self.envsize))
        self.victims_visited_sparse = set()
        
        
        # self.start_position = {'x': -2185.5, 'y': 28.0, 'z': 167.5}
        self.current_position = (self.range_z//2, self.range_x//2)
        self.relative_position = {'y':self.range_y//2, 'z':self.range_z//2, 'x':self.range_x//2}
        self.absolute_position = None
        # NOTE that we start from 0 value of x and half value for z for recording into the array
        
        # Populate with `observe()` function
        self.grid = None
        self.ypos = None
        self.zpos = None
        self.xpos = None
        self.yaw = None
        self.pitch = None
        self.lineOfSight = None
        
        self.masked_grid = None
        self.relative_map = None
        self.absolute_map = np.zeros((self.range_y, self.envsize, self.envsize))
        self.origin_coord = {'y': 27.0, 'z': 142.5, 'x': -2190.5}

        self.maze_map_dict = {}

    def get_masked_frontier_matrix(self, frontier_list, mask):
        frontier_map = np.zeros((self.envsize, self.envsize)) 
        # Z = len(frontier_list)
        for idx, coord in enumerate(frontier_list):
            if mask[idx]:
            # For now all frontiers are equally likely
                frontier_map[coord[0]][coord[1]] = 1.
        return frontier_map


    def get_directional_frontiers_mask(self, agent_pos_z, agent_pos_x, frontier_list):
        rel_frontiers = np.array(frontier_list) - np.array([agent_pos_z, agent_pos_x]).reshape(1, 2)
        east = np.where(rel_frontiers[:, 1] >= 0, 1, 0)
        west = np.where(rel_frontiers[:, 1] <= 0, 1, 0)
        north = np.where(rel_frontiers[:, 0] <= 0, 1, 0)
        south = np.where(rel_frontiers[:, 0] >= 0, 1, 0)
        return {'south' : south, 'west' : west, 'north' : north, 'east' : east}


    def is_opaque(self, item):
        for non_opaque_type in self.non_opaque_objects:
            if item == non_opaque_type:
                return False
        return True


    def inview2D_with_opaque_objects(self, grid, yaw, distance, angle=60):
        # envsize should be same as the grid size.
        # Assuming that the player has a headlight with them
        # This function is independent of environment's visibility
        envsize = grid.shape[0]
        visible_grid = np.zeros((envsize, envsize), dtype=np.uint8) 
        agent_pos = {'z': envsize//2, 'x':envsize//2}
        for theta_in_deg in range(int(yaw-angle), int(yaw+angle)):
            # print('theta_in_deg', theta_in_deg)
            for r in range(1, int(distance)):
                theta_in_rad = np.pi*theta_in_deg/180
                z = int(r*np.cos(theta_in_rad)) + agent_pos['z']
                x = - int(r*np.sin(theta_in_rad)) + agent_pos['x']
                # debug
                # print('grid[z][x]', z, x, grid[z][x])
                if z > 0 and z < envsize and x > 0 and x < envsize:
                    if self.is_opaque(grid[z][x]):
                        visible_grid[z][x] = 1
                        break
                    else:
                        visible_grid[z][x] = 1
        return visible_grid


    def get_intent_matrix(self, frontier_list, intent_likelihood):
        intent_matrix = -1*np.ones((self.envsize, self.envsize))
        for index, frontier in enumerate(frontier_list):
            intent_matrix[frontier[0]][frontier[1]] = intent_likelihood[index]
        return intent_matrix 

    def get_intent_likelihood(self, agent_pos, frontier_list, frontier_path_len, yaw):
        # import ipdb; ipdb.set_trace()
        print('yaw', yaw)
        alpha = 10
        beta = 1
        intent_likelihood = []
        resolution = 120
        angle_of_attention = 20
        # NOT REQUIRED? Only needed for std)?
        frontier_paths = np.array(frontier_path_len, dtype=np.uint64)
        yaw_in_rad = np.pi*yaw/180.
        print(yaw_in_rad)
        # distance = frontier_path_len
        # frontier_path_len.argsort()
        theta_sigma, theta_mu = 8.5,  yaw  #_in_rad
        r_sigma, r_mu = 10., 1. # frontier_paths.std(), frontier_paths.min()

        coordinates = np.array(frontier_list)
        z, x = coordinates[:, 0] - agent_pos[0], coordinates[:, 1] - agent_pos[1]
        z = np.array(z, dtype=np.float32)
        x = np.array(x, dtype=np.float32)
        # r = np.sqrt(z**2 + x**2)
        r = frontier_paths
        theta = -np.arctan2(x, z) *180/np.pi
        # joint_mean = r_mu + theta_mu
        # joint_sigma = r_sigma + theta_sigma
        frontier_yaw = (theta - theta_mu)
        print('theta', theta)
        print('frontier_yaw', frontier_yaw)
        # gtheta_raw = 1/(np.sqrt(2*np.pi) * theta_sigma) * np.exp(-((frontier_yaw)**2 / ( 2.0 * theta_sigma**2 ) ) )
        gtheta_raw = alpha * np.exp(-((frontier_yaw)**2 / ( 2.0 * theta_sigma**2 ) ) )
        # inview_condition = np.where(theta > theta_mu - angle_of_attention, 1., 0.) + np.where(theta < theta_mu + angle_of_attention, 1., 0.)
        # gtheta = gtheta_raw * inview_condition
        gtheta = gtheta_raw
        # gr = 1/(np.sqrt(2*np.pi) * r_sigma) * np.exp(-((r-r_mu)**2 / ( 2.0 * r_sigma**2 ) ) )
        gr = beta * np.exp(-((r-r_mu)**2 / ( 2.0 * r_sigma**2 ) ) )
        g = gr * gtheta
        # r, theta_in_deg = np.meshgrid(np.linspace(frontier_paths.min(),frontier_paths.max(),resolution), np.linspace(yaw-angle,yaw+angle,resolution), sparse=True)
        # theta = np.pi*theta_in_deg/180
        # gtheta = np.exp(-( (t  heta-theta_mu)**2 / ( 2.0 * theta_sigma**2 ) ) )
        # gr = np.exp(-( (r-r_mu)**2 / ( 2.0 * r_sigma**2 ) ) )
        # z = (r*np.cos(theta)) + agent_pos['z']
        # x = -(r*np.sin(theta)) + agent_pos['x'] 
        # g = gr * gtheta

        # self.prob_matrix = np.zeros((envsize, envsize))
        # for i in range(resolution):
        #     for j in range(resolution):
        #         self.prob_matrix[int(z[i][j])][int(x[i][j])] = g[i][j]

        return g



    def get_frontier_path_length(self, path_matrix, frontier_list):
        path_len = []
        for coord in frontier_list:
            path_len.append(path_matrix[coord[0], coord[1]])
        return path_len


    def get_path_matrix(self, agent_pos, absolute_map):
        # import ipdb; ipdb.set_trace()
        path_matrix = np.zeros((self.envsize, self.envsize), dtype=np.int32)
        path_matrix[agent_pos[0], agent_pos[1]] = 1
        queue = [[agent_pos[0], agent_pos[1]]]

        while len(queue):
            coordinate = queue.pop(0)           
            # print('coordinate', coordinate)
            coord_z, coord_x  = coordinate
            # print('coord_z', coord_z, 'coord_x', coord_x)

            for diff in [-1, 1]:                
                if self.is_passable_coordinate(absolute_map, coord_z + diff, coord_x):
                    # if path_matrix[coord_z + diff][coord_x] == 0:
                    if not (path_matrix[coord_z + diff][coord_x] or 0):
                        path_matrix[coord_z + diff][coord_x] = path_matrix[coord_z][coord_x] + 1
                        queue.append([coord_z + diff, coord_x])
                
                if self.is_passable_coordinate(absolute_map, coord_z, coord_x + diff):   
                    # if path_matrix[coord_z][coord_x + diff] == 0:
                    if not (path_matrix[coord_z][coord_x + diff] or 0):
                        path_matrix[coord_z][coord_x + diff] = path_matrix[coord_z][coord_x] + 1
                        queue.append([coord_z, coord_x + diff])

        return path_matrix


    def are_all_neighbours_visible(self, visibility_map, i, j):
        if i > 1 and i < self.envsize - 1 and j > 1 and j < self.envsize - 1:
            for d in [-1, 1]:
                if visibility_map[i+d][j] == 0:
                    return False
                if visibility_map[i][j+d] == 0:
                    return False
            # if OBJECT_TO_IDX['unseen']==absolute_map[i+d, j]:
            #     return False
            # if OBJECT_TO_IDX['unseen']==absolute_map[i, j+d]:
            #     return False
        return True


    def get_frontier_matrix(self, frontier_list):
        frontier_map = np.zeros((self.envsize, self.envsize)) 
        # Z = len(frontier_list)
        for coord in frontier_list:
            # For now all frontiers are equally likely
            frontier_map[coord[0]][coord[1]] = 1.
        return frontier_map


    def get_absolute_frontier_coordinates(self, masked_grid):
        frontiers = []
        for i in range(masked_grid.shape[1]):
            for j in range(masked_grid.shape[2]):
                # if masked_grid[i][j] or 0:
                if self.is_passable_coordinate(masked_grid, i, j):
                    if not self.are_all_neighbours_visible(masked_grid[1], i, j):
                        frontiers.append([i, j])
        return frontiers


    def aggregate_layerviews_for_minigrid(self):
        # RUN ONLY if self.absolute_map is populated correctly
        # Extract the passable region from below map        
        condition_list = [self.absolute_map[0,:,:]==self.object_to_index[i] for i in self.floor_objects_types]
        # import ipdb; ipdb.set_trace()
        condition = np.array(condition_list).any(axis=0)
        passable_region = np.where(condition, 1., 0.)

        # Extract the levers 
        lever_included_positions =  np.where(self.absolute_map[2,:,:]==self.object_to_index['lever'], 
            self.absolute_map[2,:,:], self.absolute_map[1,:,:])
        # TODO: Extract obstacle difficulty from above map layout
        # _, range_z, range_x = self.absolute_map.shape
        # for i in range(range_z):
        #     for j in range(range_x):
        #         if self.absolute_map[1, i, j]
        # Use the same level map mostly
        minigrid_map = lever_included_positions * passable_region
            # self.absolute_map
        return minigrid_map


    def mark_emergency_exit_visited(self):
        emergency_exit_coor = {'z': 167.5, 'x': -2192.5}
        mz = int(emergency_exit_coor['z']-self.origin_coord['z'])
        mx = int(emergency_exit_coor['x'] - self.origin_coord['x'])
        self.victims_visited[mz][mx] = 1
        self.victims_visited_sparse.add((mz, mx))


    def calc_matrix_coord(self, relative_ZPos, relative_XPos):
        z_index = relative_ZPos - self.origin_coord['z']
        x_index = relative_XPos - self.origin_coord['x']
        return z_index, x_index

    def get_client_pool(self):
        my_client_pool = MalmoPython.ClientPool()
        my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))
        # my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 20000))
        # my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10002))
        # my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10003))
        return my_client_pool



    def safeAttemptToStart(self): #, agent_host, my_mission, my_mission_record):
        # Attempt to start a mission:
        max_retries = 3
        for retry in range(max_retries):
            try:
                self.agent_host.startMission(self.my_mission, self.get_client_pool(), self.my_mission_record, 0, 'observe' )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:",e)
                    exit(1)
                else:
                    time.sleep(2)


    def safeWaitToStart(self):  
        print("Waiting for the mission to start ", end=' ')
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:",error.text)


    def observe(self):
        blended_image = None
        world_state = self.agent_host.getWorldState()
        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text

            obs = json.loads(msg)
            # with open("out.json","w") as fw:
            #     fw.write(obs)
            # import ipdb; ipdb.set_trace()
            ## DEBUG: Use ObservationFromFullStats and ObservationFromRay
            ## in mission_xml to include desired keys. Check if the keys there
            # print(obs.keys()) 
            # import ipdb; ipdb.set_trace()

            # nearby_entities = obs.get(u'entities', 0)
            # print(nearby_entities)

            # GPS-like sensor
            self.xpos = obs.get(u'XPos', 0)  # Position in 2D plane, 1st axis
            self.zpos = obs.get(u'ZPos', 0)  # Position in 2D plane, 2nd axis (yes Z!)
            self.ypos = obs.get(u'YPos', 0)  # Height as measured from surface! (yes Y!)  

            # self.absolute_position = {
            #     'y': self.ypos - self.origin_coord['y'], 
            #     'z': self.zpos - self.origin_coord['z'],
            #     'x': self.xpos - self.origin_coord['x'],
            # }

            # Standard "internal" sensory inputs
            self.yaw = obs.get(u'Yaw', 0)  # Yaw
            self.pitch = obs.get(u'Pitch', 0)  # Pitch
            print('FullStats: ', self.xpos, self.ypos, self.zpos, self.yaw, self.pitch)

            self.lineOfSight = obs.get(u'LineOfSight', -1)
            print('Ray:', self.lineOfSight)

            grid_elements = obs.get(u'relative_view', 0)

            # print(grid_elements)
            grid_list = [np.uint8(self.object_to_index.get(k, -1)) for k in grid_elements]
            self.relative_map = np.array(grid_list).reshape(self.range_y, self.range_z, self.range_x)

            # Mark the position of the agent
            # self.relative_map[:, self.current_position[0], self.current_position[1]] = 1
            # self.relative_map[:,self.relative_position['z'], self.relative_position['x']] = 1
            # inview_grid = inview(self.envsize, self.yaw, self.lineOfSight['distance'], angle=60)

            # # print(inview_grid)
            # self.visible_grid = self.inview2D_with_opaque_objects(self.relative_map[1], self.yaw, self.lineOfSight['distance'], self.angle)            
            self.visible_grid = self.inview2D_with_opaque_objects(self.relative_map[1], self.yaw, self.envsize, self.angle)            
            # # Marking agent position with 8 in the centre of the grid
            # self.visible_grid[self.relative_position['z'], self.relative_position['x']] = 8

            self.masked_grid = self.relative_map * self.visible_grid
            print('masked_grid', self.masked_grid.dtype)
            # Populating the absolute map from current view
            my = int(self.ypos - self.origin_coord['y'])
            mz = int(self.zpos - self.origin_coord['z'])
            mx = int(self.xpos - self.origin_coord['x'])

            self.absolute_position = {'y': my, 'z': mz, 'x': mx}

            for iy in range(self.sight['y'][0], self.sight['y'][1]+1):
                for iz in range(self.sight['z'][0], self.sight['z'][1]+1):
                    for ix in range(self.sight['x'][0], self.sight['x'][1]+1):
                        ay = my + iy
                        az = mz + iz
                        ax = mx + ix
                        if ay >= 0 and az >= 0 and ax >= 0 and ax < self.envsize and az < self.envsize and ay < self.range_y:
                            # if self.absolute_map[ay, az, ax] == 1: 
                            #     self.absolute_map[ay, az, ax] = 0.5
                            relative_visible_update = self.masked_grid[self.relative_position['y'] + iy, self.relative_position['z']+ iz, self.relative_position['x'] + ix] 
                            if relative_visible_update != 0:
                                self.absolute_map[ay, az, ax] = relative_visible_update
                            
                            # self.absolute_map[ay, az, ax] = self.relative_map[self.relative_position['y'] + iy, self.relative_position['z']+ iz, self.relative_position['x'] + ix]


            # self.map_image = np.array([[[map_grid_custom_colors[val] for val in row_x] for row_x in z_x_grid] for z_x_grid in self.relative_map], dtype='B')
            # plot2D(self.visible_grid, self.visible_grid_custom_colors, name='run_{}/playerFOV_angle_{}_yaw_{}_zpos_{}_xpos_{}'.format(run, angle, int(self.yaw), int(self.zpos), int(self.xpos)))
            # self.visible_grid_image = np.array([[visible_grid_custom_colors[val] for val in row] for row in self.visible_grid], dtype='B')
            
            
            ## Custom Displays for debugging: 
            # blended_image = map_image 
            # blended_image = np.stack([visible_grid_image]*self.range_y, axis=0) #* 0.5 + visible_grid_image * 0.5
            # blended_image = np.array([[[map_grid_custom_colors[val] for val in row_x] for row_x in z_x_grid] for z_x_grid in masked_grid], dtype='B')
            blended_image = np.array([[[map_grid_custom_colors[int(val)] for val in row_x] for row_x in z_x_grid] for z_x_grid in self.absolute_map], dtype='B')

            # mask = 0.45
            # blended_image = mask * map_image + (1-mask) * np.stack([visible_grid_image]*self.range_y, axis=0)
            blended_image = blended_image.astype(int)
            # self.grid = self.relative_map * visible_grid

        # time.sleep(.5)
        # pass
        return blended_image

 
    def dist_func(self, a, b, mode='l1'):
        dist = 0
        if mode == 'l2':
            for t1, t2 in zip(a, b):
                dist += (t1 - t2)**2
            return np.sqrt(dist)
        elif mode == 'l1':
            for t1, t2 in zip(a, b):
                dist += np.abs(t1 - t2)
            return np.sqrt(dist)
        else:
            raise NotImplementedError


    def explore(self):
        # What if two goals appear in the observation_from_grid? select the first unvisited one?
        # explore and astar update the belief and visited array
        # explore and astar should alternate 
        # explore can be to maximum the entropy over the belief array,(and get more 1's in visited array)
        # explore can be naively moving in positive x direction until 
        # any object of interest is found, switch to astar for it?
        pass


    def is_passable_coordinate(self, grid, z, x):
        # # Hole on the floor; can't pass
        # if self.is_passable_object(grid[0, z, x]):
        #     return False

        # same level or above is passable; then can pass
        if z > 0 and x > 0 and z < self.envsize and x < self.envsize:
            if self.is_passable_object(grid[1, z, x]) or self.is_passable_object(grid[2, z, x]):
                return True
        return False


    def is_passable_object(self, grid_item):
        # if grid_item in [9, 5, 2, 6]:  # ['air', 'fire', 'wooden_door'] 
        for item_name in self.passable_objects:
            if grid_item == self.object_to_index[item_name]:
                # print(grid_item, item_name)
                return True
        return False


    def get_passable_neighbours(self, grid, current_z, current_x):
        passable_neighbours = {}
        for dx in [-1, 1]:
            if current_x + dx > 0 and current_x + dx < grid.shape[1]:
                if self.is_passable_coordinate(grid, current_z, current_x + dx):
                    passable_neighbours[self.state_to_string(current_z, current_x + dx)] = 1

        for dz in [-1, 1]:
            if current_z + dz > 0 and current_z + dz < grid.shape[0]:
                if self.is_passable_coordinate(grid, current_z + dz, current_x):
                    passable_neighbours[self.state_to_string(current_z + dz, current_x)] = 1
        
        return passable_neighbours


    def state_to_string(self, x, y):
        return "S_%s_%s" % (str(x), str(y))

    # def plan(self, grid_2D, agent_position, goal_position, save_maze=True):
        
    #     # current_position = {'z': int(self.zpos - self.origin_coord['z']), 'x': int(self.xpos - self.origin_coord['x'])}
    #     # print('current_position', current_position)
        
    #     frontier = []


    #     maze_map_dict = {}
    #     for z in range(agent_position[0] - self.scanning_range, agent_position[0] + self.scanning_range + 1):  # 0, self.envsize):
    #         for x in range(agent_position[1] - self.scanning_range, agent_position[1] + self.scanning_range + 1): 
    #             if x >= 0 and x < self.envsize and z >= 0 and z < self.envsize:
    #                 item = grid_2D[z][x]

    #                 maze_map_dict[self.state_to_string(z, x)] = self.get_passable_neighbours(grid_2D, z, x)
                    
    #     print(maze_map_dict[self.state_to_string(*(agent_position))])
    #     print(maze_map_dict[self.state_to_string(*(goal_position))])
    #     # import ipdb; ipdb.set_trace() 

    #     maze_map = search.UndirectedGraph(maze_map_dict)

    #     # initial_position = (self.range_z//2, self.range_x//2)

    #     maze_problem = search.GraphProblem(self.state_to_string(*(agent_position)), 
    #         self.state_to_string(*(goal_position)), maze_map)
    #     print('maze_map', maze_map)
    #     solution_node = search.uniform_cost_search(problem=maze_problem, display=True) #, h=None)
    #     print('solution_node', solution_node)

    #     if solution_node is not None:
    #         solution_path = [solution_node.state]
    #         current_node = solution_node.parent
    #         solution_path.append(current_node.state)
    #         while current_node.state != maze_problem.initial:
    #             current_node = current_node.parent
    #             solution_path.append(current_node.state)
    #     return solution_path    


    def plan_explore_n_exploit(self):
        world_state = self.agent_host.getWorldState()

        done = False

        while world_state.is_mission_running and not done:
            while goal is None:
                goal = self.explore()

            self.plan(goal)

            if self.num_doors_seen == self.total_doors:
                done = True
            if self.num_victims_seen == self.total_victims:
                done = True


    def get_nearest(self, close_by_items_list, agent_position):
        minimum_dist = np.inf
        minimum_coord = None
        for coord in close_by_items_list:
            dist = self.dist_func(agent_position, coord)
            if minimum_dist > dist:
                minimum_dist = dist
                minimum_coord = coord
        return minimum_coord


    def get_agent_host(self):
        return self.agent_host



    def remove_visited(self, close_by_items_list):
        unvisited = []
        for coord_z, coord_x in close_by_items_list:
            # az = self.zpos - (self.current_position[0] - coord[0])
            # ax = self.xpos - (self.current_position[1] - coord[1])
            # z_index, x_index = self.calc_matrix_coord(az, ax)
            # print(z_index, x_index)
            if self.victims_visited[coord_z][coord_x] == 0:
                unvisited.append([coord_z, coord_x])
        return unvisited


    def check_agent_near_goal(self, agent_position, goal_position):
        for dz in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if agent_position[0] + dz == goal_position[0] and agent_position[1] + dx == goal_position[1]:
                    return True
        return False


    def get_objects_of_interest(self, grid, item_name='wool'):
        close_by_items_list = []
        # for z in range(self.range_z//2 - self.scanning_range, self.range_z//2 + self.scanning_range + 1):
            # for x in range(self.range_x//2 - self.scanning_range, self.range_x//2 + self.scanning_range + 1):
        for z in range(0, grid.shape[0]):
            for x in range(0, grid.shape[1]):
                item = grid[z][x]
                # DEBUG
                # print(item, z, x)
                if item == self.object_to_index[item_name]:
                    print('victim', 'at', z, x)
                    close_by_items_list.append([z,x])
        return close_by_items_list


def get_state_coord(string_state):
    state_list = string_state.split("_", 3)
    return int(state_list[1]), int(state_list[2])


def get_cardinal_action_commands(solution_path):
    action_list = []
    initial_position = get_state_coord(solution_path.pop())
    # goal_position = get_state_coord(solution_path.pop(0))
    while len(solution_path) != 0:
        current_position = get_state_coord(solution_path.pop())
        difference_z = current_position[0] - initial_position[0]
        difference_x = current_position[1] - initial_position[1]
        if difference_z == 1:
            action_list.append("movesouth 1")
        elif difference_z == -1:
            action_list.append("movenorth 1")
        else:
            # print("no move in z direction")
            pass 

        if difference_x == 1:
            action_list.append("moveeast 1")
        elif  difference_x == -1:
            action_list.append("movewest 1")
        else:
            pass
        initial_position = current_position

    # Turn for the final action instead of move
    # last_action = action_list.pop()
    # difference_z = current_position[0] - goal_position[0]
    # difference_x = current_position[1] - goal_position[1]
    # if difference_z == 1:
    #     action_list.append("")
    
    # LOOK each time into the direction if you move to it?

    return action_list






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xmlfile", help="Mission XML file path", default='resource/usar_personal.xml')
    parser.add_argument("--tile_size", type=int, help="size at which to render tiles (min: 8, typical: 32)", default=8)
    parser.add_argument("run", type=int, help="specify the run number for train/test data retrieval")
    args = parser.parse_args()
    debug = True


    def plot(env, trajectory_map, frontier_map, path_matrix, frontier_description):
        plt.clf() 
        fig, axs = plt.subplots(2, 2, figsize=(7, 8))
        fig.suptitle(frontier_description)

        plt.subplot(221)
        img = env.render('rgb_array', tile_size=args.tile_size)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title('Environment')
        
        plt.subplot(222)
        plt.imshow(trajectory_map)
        plt.xticks([])
        plt.yticks([])
        plt.title('trajectory map')

        plt.subplot(223)
        # plt.imshow(frontier_map)
        jet = cm.get_cmap('jet', 256)
        newcolors = jet(np.linspace(0, 1, 256))
        black = np.array([0,0,0,1])
        newcolors[:100, :] = black
        newcmp = ListedColormap(newcolors)
        plt.imshow(frontier_map, cmap=newcmp, vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title('frontier map')


        plt.subplot(224)
        plt.imshow(path_matrix, cmap='jet')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title('path matrix')

        fig.savefig('frontier_debug')
        # plt.text(0,0,frontier_description)
        plt.draw()
        # plt.show()
        plt.pause(0.1)

    def plot_minimal(frontier_map, path_matrix, frontier_description):
        plt.clf() 
        fig, axs = plt.subplots(2, 1, figsize=(5, 8))
        fig.suptitle(frontier_description)

        # plt.subplot(221)
        # img = env.render('rgb_array', tile_size=args.tile_size)
        # plt.imshow(img)
        # plt.xticks([])
        # plt.yticks([])
        # plt.title('Environment')
        
        # plt.subplot(222)
        # plt.imshow(trajectory_map)
        # plt.xticks([])
        # plt.yticks([])
        # plt.title('trajectory map')

        plt.subplot(211)
        # plt.imshow(frontier_map)
        plt.imshow(frontier_map, cmap='viridis', vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title('frontier map')


        plt.subplot(212)
        plt.imshow(path_matrix, cmap='jet')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title('path matrix')

        fig.savefig('frontier_debug')
        # plt.text(0,0,frontier_description)
        plt.draw()
        # plt.show()
        plt.pause(0.1)

    agent = PlanningAgent(args.xmlfile)
    agent.safeAttemptToStart()
    agent.safeWaitToStart()

    print("Mission running ", end=' ')
    
    agent_host = agent.get_agent_host()
    world_state = agent_host.getWorldState()
    
    # DEBUG : check if the sendCommand is working 
    if world_state.is_mission_running:
        agent_host.sendCommand('jump 1')

    agent.mark_emergency_exit_visited()

    
    for _ in range(2):
        image = agent.observe()
        time.sleep(.5)
        # grid = agent.absolute_map[1, :, :]
        minigrid_map = agent.aggregate_layerviews_for_minigrid()
        np.save('grid', minigrid_map)
        # plot2D(minigrid_map, map_grid_custom_colors, name='minigrid_map')
        # plot_with_custom_colors(image) 

    # FOR DEBUG : variable "count" controls the loop iterations so that it doesn't run till the mission's end
    count =  500

    # env = gym.make('MiniGrid-NumpyMapMinecraftUSAR-v0')
    # env = ViewSizeWrapper(env, agent_view_size=20)

    # env.reset()
    # env.render('rgb_array', args.tile_size)
    # print( agent.absolute_position, pz, px)
    while count and world_state.is_mission_running:
        print('count', count)
        image = agent.observe()
        if image is not None:
            minigrid_map = agent.aggregate_layerviews_for_minigrid()
            nz, nx = agent.absolute_position['z'], agent.absolute_position['x']
            print('agent_pos', nz, nx)
            image[1][nz][nx] = np.array([255, 0, 0], dtype='B')
            print('agent position', nz, nx)
            # env.agent_pos = [nx, nz]
            # env.grid.set(nx, nz, None)
            direction = int((agent.yaw/90 % 360 + 1) % 4)
            print('direction', direction)
            # env.agent_dir = direction
            frontier_list = agent.get_absolute_frontier_coordinates(agent.absolute_map)
            print('frontier_list', frontier_list)
            
            minigrid_map = agent.aggregate_layerviews_for_minigrid()

            directional_frontiers_mask = agent.get_directional_frontiers_mask(nz, nx, frontier_list)
            print('directional_frontiers', directional_frontiers_mask)
            
            # mask = directional_frontiers_mask['north']

            # frontier_list = agent.get_absolute_frontier_coordinates(episode_data['trajectory_map'])
            # subset_frontier_matrix = .get_frontier_matrix(frontier_list, episode_data['directional_frontiers'][direction])
        

            # TODO: change direction (4 discrete values: minigrid) 
            # with yaw (continuous values for 360 degrees): minecraft
            np.savez('../train_data/run_{:03d}_count_{:03d}'.format(args.run, count), 
                trajectory_map=agent.absolute_map, 
                agent_dir=direction,
                frontier_coordinates=frontier_list,
                directional_frontiers=directional_frontiers_mask)

            # import ipdb; ipdb.set_trace()
            # data = np.load('aggdirdata_{}.npz'.format(count))
            # print(data.files)
            # print(data['trajectory_map'])

            # directional_frontier_map = agent.get_masked_frontier_matrix(frontier_list, mask)

            # np.concatenate([minigrid_map, directional_frontier_map])

            # path_matrix = agent.get_path_matrix((nz, nx), agent.absolute_map) 
            # # print('path_matrix', path_matrix)
            # frontier_path_len = agent.get_frontier_path_length(path_matrix, frontier_list)
            # print('frontier_path_len', frontier_path_len)
            # # import ipdb; ipdb.set_trace()
            # intent_likelihood = agent.get_intent_likelihood([nz, nx], frontier_list, frontier_path_len, agent.yaw)
            # print('intent_likelihood', intent_likelihood)
            # # TODO: replace frontier map with intent likelihood
            # intent_matrix = agent.get_intent_matrix(frontier_list, intent_likelihood)
            # # print('intent_matrix', intent_matrix)
            
            # most_likely_frontier_index = np.array(intent_likelihood).argmax()
            # coord = frontier_list[most_likely_frontier_index]
            # frontier_description = "Most likely frontier is {} at {} with score={:.3f}".format(
            #     agent.index_to_object[agent.absolute_map[1, coord[0], coord[1]]],
            #     coord, 
            #     intent_likelihood[most_likely_frontier_index])


            # plot(env, image[1], intent_matrix, path_matrix, frontier_description)

            # print(frontier_description)


            # if  agent.lineOfSight['type']=='wool' and agent.lineOfSight['colour']=='WHITE':
            #     # int(agent.lineOfSight['y'])==28.0 and
            #     print('agent.lineOfSight', agent.lineOfSight['z'], agent.lineOfSight['x'])
            #     i = int(agent.lineOfSight['z'] - agent.origin_coord['z'])
            #     j = int(agent.lineOfSight['x'] - agent.origin_coord['x'])
            #     print('########\ntriaged position', i, j)
            #     env.put_obj(Goal('blue'), j, i)

            time.sleep(.5)

        count -= 1

    

if __name__ == "__main__":
    # grid = inview(envsize=11, yaw=90.0, distance=5.5, angle=75)
    # print(grid)
    main()