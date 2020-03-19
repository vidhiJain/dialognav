import MalmoPython
import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import search
import heapq

# import

# run = 3
# os.mkdir('run_{}'.format(run))


# object_to_index = {'air': 9, 'player':8, 'wooden_door':2, 'wool': 3, 
#             'stained_hardened_clay': 4, 'iron_block': 4, 'clay': 4, 
#              'fire':5, 'lever':6, 'stone_button': 7}

map_grid_custom_colors = { 
        -1: [200,200,200],                 
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
        1: [255, 255, 255],
        8: [0, 0, 255]
    }


def isOpaque(item):
    for non_opaque_type in [9, 8, 1, 2, 5, 6, 7]:
        if item == non_opaque_type:
            return False
    return True


def getMissionXML(mission_file):
    with open(mission_file, 'r') as f:
        print("Loading mission from %s" % mission_file)
        mission_xml = f.read()
    return mission_xml


def plot_with_custom_colors(image, name='debug_plot'):
    if image is None:
        return 
    figure = plt.figure(figsize=(10, 10))
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


    plt.draw()
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
    image = np.array([[d[int(val)] for val in row] for row in a], dtype='B')
    plt.imshow(image)
    plt.title('2D map') # for fires')
    # plt.xticks([])
    # plt.yticks([])
    # plt.draw()
    # plt.pause(5)
    plt.savefig(name)
    return image


class PlanningAgent():
    def __init__(self, xmlfile):

        self.agent_host = MalmoPython.AgentHost()
        self.my_mission = MalmoPython.MissionSpec(getMissionXML(xmlfile), True)
        self.my_mission_record = MalmoPython.MissionRecordSpec()

        self.objects_of_interest = ['stone_button', 'wooden_door', 'lever']
        self.object_to_index = {'air': 9, 'player':8, 'wooden_door':2, 'wool': 3, 
            'stained_hardened_clay': 4, 'iron_block': 4, 'clay': 4, 
             'fire':5, 'lever':6, 'stone_button': 7, 'gravel': 10}

        self.passable_objects = ['air', 'player', 'wooden_door'] #, 'lever', 'gravel']
        self.passable_objects_with_cost = {'air': 1,  'lever': 1, 'wooden_door': 2, 'gravel': 5}
        self.envsize = 61
        # Env specific variables; (modify them wrt xmlfile)
        # self.sight= {'x': (-3, 3), 'z': (-3, 3), 'y':(-1, 1)}
        self.sight= {'x': (-30, 30), 'z': (-30, 30), 'y':(-1, 1)}
        self.angle = 75
        self.range_x = abs(self.sight['x'][1] - self.sight['x'][0]) + 1
        self.range_y = abs(self.sight['y'][1] - self.sight['y'][0]) + 1
        self.range_z = abs(self.sight['z'][1] - self.sight['z'][0]) + 1
        self.my_mission.observeGrid(self.sight['x'][0], self.sight['y'][0], self.sight['z'][0], 
            self.sight['x'][1], self.sight['y'][1], self.sight['z'][1], 'relative_view')
        self.scanning_range = 5
        
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
        self.origin_coord = {'y': 27.0, 'z': 140.5, 'x': -2195.5}

        self.maze_map_dict = {}

        self.frontier = set()

    def mark_emergency_exit_visited(self):
        emergency_exit_coor = {'z': 167.5, 'x': -2192.5}
        mz = int(emergency_exit_coor['z']-self.origin_coord['z'])
        mx = int(emergency_exit_coor['x'] - self.origin_coord['x'])
        self.victims_visited[mz][mx] = 1
        self.victims_visited_sparse.add((mz, mx))




    def inview2D_with_opaque_objects(self, grid, yaw, distance, angle=60):
        # envsize should be same as the grid size.
        # Assuming that the player has a headlight with them
        # This function is independent of environment's visibility
        # Modifying this function to store the frontier nodes
        envsize = grid.shape[0]
        visible_grid = np.zeros((envsize, envsize)) 
        agent_pos = {'z': envsize//2, 'x':envsize//2}
        for theta_in_deg in range(int(yaw-angle), int(yaw+angle)):
            # print('theta_in_deg', theta_in_deg)
            for r in range(1, int(distance)+1):
                theta_in_rad = np.pi*theta_in_deg/180
                z = int(r*np.cos(theta_in_rad)) + agent_pos['z'] 
                x = - int(r*np.sin(theta_in_rad)) + agent_pos['x']
                # debug
                # print('grid[z][x]', z, x, grid[z][x])
                # if theta_in_deg == int(yaw-angle) or theta_in_deg == int(yaw+angle):
                #     if self.is_passable(grid[z][x]):
                #         self.frontier.add((z,x))
                # else:
                #     self.frontier.discard((z,x))

                if isOpaque(grid[z][x]):
                    visible_grid[z][x] = 1
                    # if self.is_passable(grid[z][x]):
                    #     self.frontier.add((z,x))
                    break
                else:
                    visible_grid[z][x] = 1
                    # self.frontier.add((z,x))
        print('frontier', self.frontier)
        return visible_grid


    def explore(self):
        # What if two goals appear in the observation_from_grid? select the first unvisited one?
        # explore and astar update the belief and visited array
        # explore and astar should alternate 
        # explore can be to maximum the entropy over the belief array,(and get more 1's in visited array)
        # explore can be naively moving in positive x direction until 
        # any object of interest is found, switch to astar for it?
        self.aggregate_frontiers()
        print(self.frontier)
        if self.frontier is not None:
            current_goal = self.frontier.pop() 
        else:
            print("No frontier found?")
        # self.plan()
        return current_goal


    def any_neighbours_not_visible(self, grid, y, z, x):
        # self.absolute_map[y, z+, x]
        # passable_neighbours = {}
        for dx in [-1, 1]:
            if x + dx > 0 and x + dx < grid.shape[1]:
                if grid[y, z, x+dx] == 0:  # not visible
                    return True
        for dz in [-1, 1]:
            if z + dz > 0 and z + dz < grid.shape[0]:
                if grid[y, z+dz, x] == 0:  # not visible
                    return True        
        return False


    def aggregate_frontiers(self, y=1):
        for z in range(self.envsize):
            for x in range(self.envsize):
                if self.absolute_map[y, z, x] != 0: # is visible
                    if self.any_neighbours_not_visible(self.absolute_map,y,z,x):
                        if self.is_passable(self.absolute_map[y, z, x]):
                            self.frontier.add((z,x))
                    # else:
                        # self.frontier.discard((z,x))



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

            # Standard "internal" sensory inputs
            self.yaw = obs.get(u'Yaw', 0)  # Yaw
            self.pitch = obs.get(u'Pitch', 0)  # Pitch
            # print('FullStats: ', self.xpos, self.ypos, self.zpos, self.yaw, self.pitch)

            self.lineOfSight = obs.get(u'LineOfSight', -1)
            # print('Ray:', self.lineOfSight)

            grid_elements = obs.get(u'relative_view', 0)
            # print(grid_elements)
            grid_list = [self.object_to_index.get(k, -1) for k in grid_elements]
            self.relative_map = np.array(grid_list).reshape(self.range_y, self.range_z, self.range_x)

            # Mark the position of the agent
            # self.relative_map[:, self.current_position[0], self.current_position[1]] = 1
            self.relative_map[:,self.relative_position['z'], self.relative_position['x']] = 8
            # inview_grid = inview(self.envsize, self.yaw, self.lineOfSight['distance'], angle=60)

            # # print(inview_grid)
            self.visible_grid = self.inview2D_with_opaque_objects(self.relative_map[1], self.yaw, self.lineOfSight['distance'], self.angle)            
            # print(self.frontier)
            # # Marking agent position with 8 in the centre of the grid
            self.visible_grid[self.relative_position['z'], self.relative_position['x']] = 1

            self.masked_grid = self.relative_map * self.visible_grid

            # Finding frontiers?
            # self.finding_frontiers()
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
                            if self.masked_grid[self.relative_position['y'] + iy, self.relative_position['z']+ iz, self.relative_position['x'] + ix] != 0:
                                self.absolute_map[ay, az, ax] = self.masked_grid[self.relative_position['y'] + iy, self.relative_position['z']+ iz, self.relative_position['x'] + ix] 
                            # self.absolute_map[ay, az, ax] = self.relative_map[self.relative_position['y'] + iy, self.relative_position['z']+ iz, self.relative_position['x'] + ix]

            



            # self.map_image = np.array([[[map_grid_custom_colors[val] for val in row_x] for row_x in z_x_grid] for z_x_grid in self.relative_map], dtype='B')
            # plot2D(self.visible_grid, self.visible_grid_custom_colors, name='run_{}/playerFOV_angle_{}_yaw_{}_zpos_{}_xpos_{}'.format(run, angle, int(self.yaw), int(self.zpos), int(self.xpos)))
            # self.visible_grid_image = np.array([[visible_grid_custom_colors[val] for val in row] for row in self.visible_grid], dtype='B')
            
            
            ## Custom Displays for debugging: 
            # blended_image = map_image 
            # blended_image = np.stack([visible_grid_image]*self.range_y, axis=0) #* 0.5 + visible_grid_image * 0.5
            # blended_image = np.array([[[map_grid_custom_colors[val] for val in row_x] for row_x in z_x_grid] for z_x_grid in masked_grid], dtype='B')
            blended_image = np.array([[[map_grid_custom_colors[int(val)] for val in row_x] for row_x in z_x_grid] for z_x_grid in self.absolute_map], dtype='B')
            # print(blended_image.shape)
            # mask = 0.45
            # blended_image = mask * map_image + (1-mask) * np.stack([visible_grid_image]*self.range_y, axis=0)
            blended_image = blended_image.astype(int)
            # self.grid = self.relative_map * visible_grid

        time.sleep(.5)
        # pass
        return blended_image

    
    # def finding_frontiers(self, masked_grid):
    #     # finding_frontiers

        



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


    # def explore(self):
    #     # What if two goals appear in the observation_from_grid? select the first unvisited one?
    #     # explore and astar update the belief and visited array
    #     # explore and astar should alternate 
    #     # explore can be to maximum the entropy over the belief array,(and get more 1's in visited array)
    #     # explore can be naively moving in positive x direction until 
    #     # any object of interest is found, switch to astar for it?
    #     print(self.frontier)
    #     if self.frontier:
    #         current_goal = self.frontier.pop()  # Ideally, heapify and do self.frontier.pop() 
    #     else:
    #         print("No frontier found?")
    #     # self.plan()
    #     return current_goal



    def is_passable(self, grid_item):
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
                if self.is_passable(grid[current_z][current_x + dx]):
                    passable_neighbours[self.state_to_string(current_z, current_x + dx)] = 1

        for dz in [-1, 1]:
            if current_z + dz > 0 and current_z + dz < grid.shape[0]:
                if self.is_passable(grid[current_z + dz][current_x]):
                    passable_neighbours[self.state_to_string(current_z + dz, current_x)] = 1
        
        return passable_neighbours


    def state_to_string(self, x, y):
        return "S_%s_%s" % (str(x), str(y))

    def plan(self, grid_2D, agent_position, goal_position, save_maze=True):
        
        # current_position = {'z': int(self.zpos - self.origin_coord['z']), 'x': int(self.xpos - self.origin_coord['x'])}
        # print('current_position', current_position)
        
        # frontier = []


        maze_map_dict = {}
        for z in range(agent_position[0] - self.scanning_range, agent_position[0] + self.scanning_range + 1):  # 0, self.envsize):
            for x in range(agent_position[1] - self.scanning_range, agent_position[1] + self.scanning_range + 1): 
                if x >= 0 and x < self.envsize and z >= 0 and z < self.envsize:
                    item = grid_2D[z][x]
                    # if maze_map_dict.get(self.state_to_string(z, x)) is None:
                    #     maze_map_dict[self.state_to_string(z, x)] = set()
                    # maze_map_dict[self.state_to_string(z, x)].add(self.get_passable_neighbours(grid_2D, z, x))        import ipdb; ipdb.set_trace()
                    # import ipdb; ipdb.set_trace()

                    if self.is_passable(item):
                        maze_map_dict[self.state_to_string(z, x)] = self.get_passable_neighbours(grid_2D, z, x)
                    
        # print(maze_map_dict)

        print('neighbours to agent_position', maze_map_dict[self.state_to_string(*(agent_position))])
        print('neighbours to goal_position', maze_map_dict[self.state_to_string(*(goal_position))])
        # import ipdb; ipdb.set_trace() 

        maze_map = search.UndirectedGraph(maze_map_dict)

        # initial_position = (self.range_z//2, self.range_x//2)

        maze_problem = search.GraphProblem(self.state_to_string(*(agent_position)), 
            self.state_to_string(*(goal_position)), maze_map)
        # print('maze_map', maze_map)
        solution_node = search.uniform_cost_search(problem=maze_problem, display=True) #, h=None)
        print('solution_node', solution_node)

        if solution_node is not None:
            solution_path = [solution_node.state]
            current_node = solution_node.parent
            solution_path.append(current_node.state)
            while current_node.state != maze_problem.initial:
                current_node = current_node.parent
                solution_path.append(current_node.state)
        return solution_path    


    # def plan_in_relative_grid(self, grid_2D, goal_position):
        
    #     current_position = (self.range_z//2, self.range_x//2)
    #     frontier = []


    #     maze_map_dict = {}
    #     for z in range(self.range_z//2 - self.scanning_range, self.range_z//2 + self.scanning_range + 1):
    #         for x in range(self.range_x//2 - self.scanning_range, self.range_x//2 + self.scanning_range + 1):
    #             item = grid_2D[z][x]
    #             # if self.maze_map_dict[self.state_to_string(z, x)] is None:
    #                 # self.maze_map_dict[self.state_to_string(z, x)] = set()
    #             maze_map_dict[self.state_to_string(z, x)] = self.get_passable_neighbours(grid_2D, z, x)
        
    #     # print(maze_map_dict)
    #     # import ipdb; ipdb.set_trace() 

    #     maze_map = search.UndirectedGraph(maze_map_dict)
    #     print('maze_map', maze_map)

    #     initial_position = (self.range_z//2, self.range_x//2)

    #     maze_problem = search.GraphProblem(self.state_to_string(*(initial_position)), 
    #         self.state_to_string(*(goal_position)), maze_map)

    #     solution_node = search.astar_search(problem=maze_problem, display=True) #, h=None)
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

    # def check_agent_near_goal(self, az, ax):
    #     for dz in [-1, 0, 1]:
    #         for dx in [-1, 0, 1]:
    #             if az + dz == self.zpos and ax + dx == self.xpos:
    #                 return True
    #     return False

# BELIEF and VISITED (incomplete)----------------------------
    # def update_belief(self):
    #     # Based on the movement, the belief should update about location of goals
    #     self.belief =         

    # def update_visited():
    #     # Visted array must be updated, where current position is indicated on that array
# ----------------------------


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


def get_relative_action_commands(solution_path):
    action_list = []
    initial_position = get_state_coord(solution_path.pop())
    current_yaw = self.yaw
    # goal_position = get_state_coord(solution_path.pop(0))
    while len(solution_path) != 0:
        current_position = get_state_coord(solution_path.pop())
        difference_z = current_position[0] - initial_position[0]
        difference_x = current_position[1] - initial_position[1]
        if difference_z == 1:
            # action_list.append("movesouth 1")
            intended_yaw = 0.0
        elif difference_z == -1:
            # action_list.append("movenorth 1")
            intended_yaw = 180.0
        else:
            # print("no move in z direction")
            pass 

        if difference_x == 1:
            # action_list.append("moveeast 1")
            intended_yaw = -90.0
        elif  difference_x == -1:
            # action_list.append("movewest 1")
            intended_yaw = 90.0
        else:
            pass

        yaw_difference = intended_yaw - current_yaw 
        if yaw_difference <= 180:

        action_list.append("move 1")
        initial_position = current_position

    # Turn for the final action instead of move
    # last_action = action_list.pop()
    # difference_z = current_position[0] - goal_position[0]
    # difference_x = current_position[1] - goal_position[1]
    # if difference_z == 1:
    #     action_list.append("")
    
    # LOOK each time into the direction if you move to it?

    return action_list


def create_frontier_map(map_size, frontier_list): #, agent_position):
    grid = np.zeros((map_size, map_size))
    for coord_z, coord_x in frontier_list:
        grid[coord_z, coord_x] = 1
    return grid


def main():
    debug = True

    agent = PlanningAgent('usar.xml')
    agent.safeAttemptToStart()
    agent.safeWaitToStart()

    print("Mission running ", end=' ')
    
    agent_host = agent.get_agent_host()
    world_state = agent_host.getWorldState()
    
    # DEBUG : check if the sendCommand is working 
    if world_state.is_mission_running:
        agent_host.sendCommand('jump 1')

    agent.mark_emergency_exit_visited()

    # FOR DEBUG : variable "count" controls the loop iterations so that it doesn't run till the mission's end
    count = 40

    while count and world_state.is_mission_running:
        print('count', count)
        image = agent.observe()
        if image is not None:
            # plot_with_custom_colors(image)
            map_image = plot2D(agent.absolute_map[1, :, :], map_grid_custom_colors, name='absolute_map_view')
            frontier_map = create_frontier_map(agent.envsize, agent.frontier)
            frontier_image = plot2D(frontier_map, visible_grid_custom_colors, name='frontiers_view')

            mask = 0.45
            blended_image = mask * map_image + (1-mask) * frontier_image
            plt.clf()
            plt.imshow(blended_image)
            plt.savefig('blended_image')

            # To be made into user-passed arguments
            grid = agent.absolute_map[1, :, :]
            item_name = 'wool'
            # print(grid)
            agent_position = agent.absolute_position['z'], agent.absolute_position['x']
            
            # if grid is not None:

            close_items = agent.get_objects_of_interest(grid, item_name)
            unvisited_close_items = agent.remove_visited(close_items)
            nearest_coord = agent.get_nearest(unvisited_close_items, agent_position)

            print('nearest_coord', nearest_coord)
            if nearest_coord is None:
                nearest_coord = agent.explore()
            # rz, rx = nearest_coord[0], nearest_coord[1]
            # if agent.check_agent_near_goal(rz, rx):
            #     mz, mx = agent.calc_matrix_coord(az, ax)
            #     agent.victims_visited[int(mz), int(mx)] = 1
            #     agent.victims_visited_sparse.add((int(mz), int(mx)))
            #     # break
            # else:
            if nearest_coord is not None:
                solution_path = agent.plan(grid, agent_position, nearest_coord, save_maze=True)
            #     goal_node = solution_path[0]
                print('solution_path', solution_path)

                action_list = get_cardinal_action_commands(solution_path)
                print('action_list', action_list)
                # last_action = action_list.pop()
                for action in action_list[:-1]:
                    print('executing ', action)
                    agent.observe()
                    last_zpos, last_xpos = agent.zpos, agent.xpos
                    print('last_zpos', last_zpos, 'last_xpos', last_xpos)

                    agent_host.sendCommand(action)
                    time.sleep(1)
                    image = agent.observe()
                    plot_with_custom_colors(image)
                    current_zpos, current_xpos = agent.zpos, agent.xpos
                    print('current_zpos', current_zpos, 'current_xpos', current_xpos)
                    if last_xpos == current_xpos and last_zpos == current_zpos:
                        # i = 0
                        # if i < 10:
                        #     agent.agent_host.sendCommand('attack 1')
                        #     i += 1
                        # else:
                        print('no movement happened. break!')
                        # break
                
                if agent.check_agent_near_goal(agent_position, nearest_coord) is True:
                    agent.victims_visited[nearest_coord[0], nearest_coord[1]] = 1
                    agent.victims_visited_sparse.add((int(nearest_coord[0]), int(nearest_coord[1])))
                    print('victims_visited', agent.victims_visited_sparse)

            # 
            #         time.sleep(.5)
                
            # # import ipdb;ipdb.set_trace()
            # rz, rx = get_state_coord(goal_node)
            # az, ax = agent.zpos - (agent.current_position[0] - rz) , agent.xpos - (agent.current_position[1] - rx)

            # if agent.check_agent_near_goal(rz, rx):
            #     mz, mx = agent.calc_matrix_coord(az, ax)
            #     agent.victims_visited[int(mz), int(mx)] = 1
            #     agent.victims_visited_sparse.add((int(mz), int(mx)))
            # print('victims_visited', agent.victims_visited_sparse)
            # # Finish the wool/stone_button there~
            # while agent.lineOfSight['type'] != 'wool':
            #     grid = agent.observe()
            #     if agent.pitch == 0:
            #         agent_host.sendCommand('look 1')
            #     elif agent.pitch == 90:
            #         agent_host.sendCommand('look -1')
            #     else:
            #         pass
            #     agent_host.sendCommand('turn 1')
            #     # time.sleep(1)
            #     if agent.lineOfSight['type'] == 'wool':
            #         break
            # agent_host.sendCommand('attack 1')
                
        time.sleep(.5)


# --------------------------------------------------------------------
            ## Incomplete!

            ## Wooden Door
            ## code to search and interact with the wooden door 

            # close_by_items_list = []
            # for z in range(agent.range_z//2 - agent.scanning_range, agent.range_z//2 + agent.scanning_range + 1):
            #     for x in range(agent.range_x//2 - agent.scanning_range, agent.range_x//2 + agent.scanning_range + 1):
            #         item = grid[2][z][x]
            #         print(item, z, x)
            #         if item == agent.object_to_index['wooden_door']:
            #             print('wooden_door', 'at', z, x)
            #             close_by_items_list.append([z,x])

            # nearest_coord = agent.get_nearest(close_by_items_list)
            # print(nearest_coord)

            # solution_path = agent.plan(grid[2], nearest_coord)
            
            # print(solution_path) 
            # if agent.lineOfSight['type'] != 'wooden_door':
            #     agent_host.sendCommand('turn 1')
            
            # if agent.lineOfSight['type'] == 'wooden_door' and not agent.lineOfSight['open']:
            #     agent_host.sendCommand('use 1')

            ## Lever

            # while agent.lineOfSight['type'] != 'lever':
            #     agent_host.sendCommand('turn 1')
            

            # ## Use command is not working!!!
            # if agent.lineOfSight['type'] == 'lever': # and agent.lineOfSight['prop_powered']:
            #     print("agent_host.sendCommand('attack 1')")
            #     print(agent.lineOfSight['prop_powered'])
            #     agent_host.sendCommand('attack 1') 
            #     time.sleep(1)
# --------------------------------------------------------------------



        count -= 1

            

if __name__ == "__main__":
    # grid = inview(envsize=11, yaw=90.0, distance=5.5, angle=75)
    # print(grid)
    main()