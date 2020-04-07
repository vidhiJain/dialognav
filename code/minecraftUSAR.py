import MalmoPython
import numpy as np
import time
import json
import os


def getMissionXML(mission_file):
    with open(mission_file, 'r') as f:
        print("Loading mission from %s" % mission_file)
        mission_xml = f.read()
    return mission_xml


class MinecraftUSAR:
    """
    To observe the USAR environment in `usar.xml` 

    Note 1: player specific variables per timestep: 
        xpos, ypos, zpos, yaw, pitch, lineOfSight, observationFromGrid(relative)
    
    Note 2: based on the mission xml, set the properties in the constructor below. 
    For example, envsize, angle for FOV, passable object types, opaque object types, etc.
    """
    def __init__(self, xmlfile, recordingsDirectory="../human_trajectories"):

        self.agent_host = MalmoPython.AgentHost()
        self.my_mission = MalmoPython.MissionSpec(getMissionXML(xmlfile), True)
        self.my_mission_record = MalmoPython.MissionRecordSpec()

        # self.recordingsDirectory = recordingsDirectory
        # if(not os.path.exists(recordingsDirectory)):
        #     os.mkdir(recordingsDirectory)

        # if recordingsDirectory:
        #     self.my_mission_record.recordRewards()
        #     self.my_mission_record.recordObservations()
        #     self.my_mission_record.recordCommands()
        #     # if agent_host.receivedArgument("record_video"): # my_mission_record.recordMP4(24,2000000)
        #     self.my_mission_record.recordMP4(24,2000000)
        # recordingsDirectory = malmoutils.get_recordings_directory(agent_host)

        # Should not be here - move to agent
        self.priority = {'stone_button': 70, 'wooden_door': 40, 'lever': 20}
        self.objects_of_interest = ['stone_button', 'wooden_door', 'lever']
        self.object_to_index = {
            'air': 9, 
            'player': 8, 
            'wooden_door': 2, 
            'wool': 3, 
            'stained_hardened_clay': 4, 'iron_block': 4, 'clay': 4, 
            'fire': 5, 
            'lever': 6, 
            'stone_button': 7, 
            'gravel': 10
        }


        self.passable_objects = ['air'] #, 'wooden_door'] #, 'lever', 'gravel']
        self.passable_objects_with_cost = {'air': 1,  'lever': 1, 'wooden_door': 2, 'gravel': 5}
        self.non_opaque_list = ['air', 'player', 'wooden_door', 'fire', 'lever', 'stone_button']  # [9, 8, 1, 2, 5, 6, 7]
        
        # Env specific variables; (modify them wrt xmlfile)
        # self.sight= {'x': (-3, 3), 'z': (-3, 3), 'y':(-1, 1)}
        self.envsize = 61
        self.angle = 51     # for field of view imposition on (raw) observation from grid
        self.scanning_range = 15
        # self.pitch = np.ran
        self.origin_coord = {'y': 27.0, 'z': 140.5, 'x': -2195.5}

        self.sight= {'x': (-30, 30), 'z': (-30, 30), 'y':(-1, 1)}

        self.range_x = abs(self.sight['x'][1] - self.sight['x'][0]) + 1
        self.range_y = abs(self.sight['y'][1] - self.sight['y'][0]) + 1
        self.range_z = abs(self.sight['z'][1] - self.sight['z'][0]) + 1
        self.my_mission.observeGrid(self.sight['x'][0], self.sight['y'][0], self.sight['z'][0], 
            self.sight['x'][1], self.sight['y'][1], self.sight['z'][1], 'relative_view')
        
        # self.start_position = {'x': -2185.5, 'y': 28.0, 'z': 167.5}
        self.current_position = (self.range_z//2, self.range_x//2)
        self.relative_position = {'y':self.range_y//2, 'z':self.range_z//2, 'x':self.range_x//2}
        self.absolute_position = None
        # NOTE that we start from 0 value of x and half value for z for recording into the array
        
        # Populate with `observe()` function
        # self.grid = None
        # self.ypos = None
        # self.zpos = None
        # self.xpos = None
        # self.yaw = None
        # self.pitch = None
        # self.lineOfSight = None
        
        # self.masked_grid = None
        relative_map = None
        self.absolute_map = np.zeros((self.range_y, self.envsize, self.envsize))

        self.maze_map_dict = {}


        # (TODO: shift to the planning agent class>?' 
        # Goal specific variables
        self.num_victims_seen = 0
        self.num_doors_seen = 0
        self.total_victims = 3
        self.total_doors = 3
        self.victims_visited = np.zeros((self.envsize, self.envsize))
        self.victims_visited_sparse = set()


    """
    Malmo specific function
    """
    def get_recording_frame(self, img_counter=0):
        """
        Refer collect_human_videos.py in runner
        """

        while world_state.number_of_video_frames_since_last_state < 1 and world_state.is_mission_running:
            logger.info("Waiting for frames...")
            time.sleep(0.05)
            world_state = agent_host.getWorldState()
        
        if world_state.is_mission_running:
            frame = world_state.video_frames[-1]
            img = Image.frombytes('RGB', (640,480), bytes(frame.pixels))
            imageio.imsave("./tmp_imgs/{}.png".format(img_counter), img)
            img_counter += 1


    def get_agent_host(self):
        return self.agent_host

    def get_client_pool(self):
        """ 
        Malmo specific function: To create client pool for connecting to the minecraft server
        """
        my_client_pool = MalmoPython.ClientPool()
        my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))
        # my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 20000))
        # my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10002))
        # my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10003))
        return my_client_pool


    def safeAttemptToStart(self): #, agent_host, my_mission, my_mission_record):
        """
        Malmo specific function# Attempt to start a mission:
        """
    
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
        """
        Malmo specific function# Attempt to start a mission:
        """
        print("Waiting for the mission to start ", end=' ')
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:",error.text)


    def create_maze_adjacency_list(self, grid_2D):
        """
        TODO: 
            1. Make it 3D graph
            2. Allow all objects to be passable 
                with different costs
        """

        maze_map_dict = {}
        # for z in range(agent_position[0] - obs.scanning_range, agent_position[0] + obs.scanning_range + 1):  # 0, obs.envsize):
        for z in range(grid_2D.shape[0]):
            # for x in range(agent_position[1] - obs.scanning_range, agent_position[1] + obs.scanning_range + 1): 
            for x in range(grid_2D.shape[1]):

                if x >= 0 and x < self.envsize and z >= 0 and z < self.envsize:
                    item = grid_2D[z][x]
                    # if maze_map_dict.get(obs.state_to_string(z, x)) is None:
                    #     maze_map_dict[obs.state_to_string(z, x)] = set()
                    # maze_map_dict[obs.state_to_string(z, x)].add(obs.get_passable_neighbours(grid_2D, z, x))        import ipdb; ipdb.set_trace()
                    # import ipdb; ipdb.set_trace()
                    if self.is_passable(item):
                        maze_map_dict[self.get_2Dstate_to_string(z, x)] = self.get_passable_neighbours(grid_2D, z, x)
        self.maze_map_dict.update(maze_map_dict)
        return self.maze_map_dict


    def observe(self):
        """
        Extracts the variables from observation to a dictionary 
        """
        blended_image = None
        world_state = self.agent_host.getWorldState()
        if world_state.number_of_observations_since_last_state > 0:
            timestamp = world_state.observations[-1].timestamp
            msg = world_state.observations[-1].text 
            obs = json.loads(msg)
            return {'timestamp': timestamp, 'observations': obs}
    

    def inview2D_with_opaque_objects(self, grid, yaw, distance, angle=60):
        """
        Using polar coordinates to return the blocks in the 
        field of view of minecraft player

        Parameters:
            the grid and player's yaw, 
            line of sight distance, and 
            angle for field of view
        Returns: 
            visible grid: 2D numpy array

        Note: 
        envsize should be same as the grid size.
        Assuming that the player has a headlight with them
        This function is independent of environment's visibility

        TODO: Extend to 3D visibility by 3D polar coordinates check
        """

        envsize = grid.shape[0]
        visible_grid = np.zeros((envsize, envsize)) 
        agent_pos = {'z': envsize//2, 'x':envsize//2}
        for theta_in_deg in range(int(yaw-angle), int(yaw+angle)):
            # print('theta_in_deg', theta_in_deg)
            for r in range(1, int(distance)):
                theta_in_rad = np.pi*theta_in_deg/180
                z = int(r*np.cos(theta_in_rad)) + agent_pos['z']
                x = - int(r*np.sin(theta_in_rad)) + agent_pos['x']
                # debug
                # print('grid[z][x]', z, x, grid[z][x])
                if self.is_opaque(grid[z][x]):
                    visible_grid[z][x] = 1
                    break
                else:
                    visible_grid[z][x] = 1
        return visible_grid        


    # def inview3D_with_opaque_objects(self, grid, yaw, pitch, distance, angle=60):
    #     """
    #     Using polar coordinates to return the blocks in the 
    #     field of view of minecraft player

    #     Parameters:
    #         the grid and player's yaw, 
    #         line of sight distance, and 
    #         angle for field of view
    #     Returns: 
    #         visible grid: 2D numpy array

    #     Note: 
    #     envsize should be same as the grid size.
    #     Assuming that the player has a headlight with them
    #     This function is independent of environment's visibility

    #     TODO: Extend to 3D visibility by 3D polar coordinates check
    #     """
        
    #     envsize = grid.shape[0]
    #     visible_grid = np.zeros((envsize, envsize)) 
    #     agent_pos = self.absolute_position
    #     # for phi_in_deg in range(int(pitch-))
    #     for theta_in_deg in range(int(yaw-angle), int(yaw+angle)):
    #         # print('theta_in_deg', theta_in_deg)
    #         for r in range(1, int(distance)):
    #             theta_in_rad = np.pi*theta_in_deg/180
    #             z = int(r*np.cos(theta_in_rad)) + agent_pos['z']
    #             x = - int(r*np.sin(theta_in_rad)) + agent_pos['x']
    #             # debug
    #             # print('grid[z][x]', z, x, grid[z][x])
    #             if self.is_opaque(grid[z][x]):
    #                 visible_grid[z][x] = 1
    #                 break
    #             else:
    #                 visible_grid[z][x] = 1
    #     return visible_grid       

    def get_relative_map(self, elements_list, player_marker=1):
        """
        To create the relative map with the player in the centre
        with the relative_elements_list 
        (from ObservationFromGrid/relative_view in obs)  

        Parameters: 
            elememt_list: list of string of object names 

        Returns:
            relative_map: 3D numpy array 
        """
        # relative_elements_list = obs.get(u'relative_view', 0)
        grid_list = [self.object_to_index.get(k, -1) for k in elements_list]
        relative_map = np.array(grid_list).reshape(self.range_y, self.range_z, self.range_x)
        # Mark the position of the agent
        relative_map[:,self.relative_position['z'], self.relative_position['x']] = player_marker

        return relative_map


    def get_visible_relative_map(self, relative_view, yaw, lineOfSight):
        """
        Applies the visibility restriction for human like FOV 
        and enforces opaqueness of objects 
        (Player's visiblity is not respected by `observationFromGrid`. 
        It gives observation through opaque objects too!)
        """
        # relative_view = obs.get(u'relative_view', 0)
        # yaw = obs.get(u'Yaw', 0)
        # lineOfSightDistance = obs.get(u'LineOfSight', {}).get('distance', -1)
        lineOfSightDistance = lineOfSight.get('distance', np.inf)
        relative_map = self.get_relative_map(relative_view, player_marker=1)
        # inview_grid = inview(self.envsize, yaw, lineOfSightDistance, angle=60)
        visible_map = self.inview2D_with_opaque_objects(relative_map[1], yaw, lineOfSightDistance, angle=60)
        # print(inview_map)
        # Marking agent position with 8 in the centre of the map
        # visible_map[self.relative_position['z'], self.relative_position['x']] = 8
        masked_map = relative_map * visible_map
        return masked_map

    def update_agent_position(self, player_coordinates):
        """
        Player coordinates are extracted from obs['XPos'], etc.
        """
        my = int(player_coordinates['y'] - self.origin_coord['y'])
        mz = int(player_coordinates['z'] - self.origin_coord['z'])
        mx = int(player_coordinates['x'] - self.origin_coord['x'])
        self.absolute_position = {'y': my, 'z': mz, 'x': mx}
        return self.absolute_position

    def get_visible_absolute_grid_map(self, visible_relative_map, player_coordinates):
        """
        Initialize the locations on the absolute map based on visible relative elements
        
        Parameters: 
            visible_relative_map: 3D numpy array 
                of player's scanning range size

        Returns:
            absolute_map: 3D numpy array 
                of actual env size specified

        """
        my = self.absolute_position['y']
        mz = self.absolute_position['z']
        mx = self.absolute_position['x']
        
        ry = self.relative_position['y']
        rz = self.relative_position['z']
        rx = self.relative_position['x']

        for iy in range(self.sight['y'][0], self.sight['y'][1]+1):
            for iz in range(self.sight['z'][0], self.sight['z'][1]+1):
                for ix in range(self.sight['x'][0], self.sight['x'][1]+1):
                    ay = my + iy
                    az = mz + iz
                    ax = mx + ix
                    if ax < self.envsize and az < self.envsize and ay < self.range_y:
                        if ay >= 0 and az >= 0 and ax >= 0:
                            if visible_relative_map[ry + iy, rz + iz, rx + ix] != 0:
                                self.absolute_map[ay, az, ax] = visible_relative_map[ry + iy, rz + iz, rx + ix] 

        return self.absolute_map


    def is_passable(self, grid_item):
        # if grid_item in [9, 5, 2, 6]:  # ['air', 'fire', 'wooden_door'] 
        for item_name in self.passable_objects:
            if grid_item == self.object_to_index[item_name]:
                print(grid_item, item_name)
                return True
        return False


    def is_opaque(self, item):
        for non_opaque_type in self.non_opaque_list:
            if item == self.object_to_index[non_opaque_type]:
                return False
        return True


    def get_passable_neighbours_2D(self, grid, current_z, current_x):
        """
        Checks the neighbours of current position in four cardinal directions, 
        and adds them to the list of neighbouring locations where the agent can navigate to.
            If current position is not passable, its neighbours should not be considered.
        Input: 
            grid: 2D numpy array, z and x coordinates     
        Returns: 
            list of passable_neighbours 
        """

        passable_neighbours = {}

        if not self.is_passable(grid[current_z][current_x]):
            return passable_neighbours

        for d in [-1, 1]:
            if current_x + d > 0 and current_x + d < grid.shape[1]:
                if self.is_passable(grid[current_z][current_x + d]):
                    passable_neighbours[self.get_2Dstate_to_string(current_z, current_x + d)] = 1

            if current_z + d > 0 and current_z + d < grid.shape[0]:
                if self.is_passable(grid[current_z + d][current_x]):
                    passable_neighbours[self.get_2Dstate_to_string(current_z + d, current_x)] = 1
        
        return passable_neighbours


    def get_passable_neighbours_3D(self, grid, y, z, x):
        """
        Same as get_passable_neighbours_2D 
        include the neighbours along y-axis (height) 
        if jump is possible

        Parameters: 
            grid: 2D numpy array, and y, z, x coordinates  
        Returns:
            list of passable neighbours 

        TODO: Ideally, the check for passable neighbour should be 
        that if a flat neighbour is passable then the block below it should be impassable.
        As we don't check for it for now, we rely on Minecraft's gravity to 
        bring the player to the lowest y coordinate passable under the flat neighbour block.
        """

        passable_neighbours = {}
        
        if not self.is_passable(grid[y][z][x]):
            return passable_neighbours


        for d in [-1, 1]:
            if x + d > 0 and x + d < grid.shape[1]:
                if self.is_passable(grid[y][z][x + d]):
                    passable_neighbours[self.get_3Dstate_to_string(y, z, x + d)] = 1
                elif self.is_passable(grid[y+1][z][x + d]):
                    passable_neighbours[self.get_3Dstate_to_string(y+1, z, x + d)] = 1

            if z + d > 0 and z + d < grid.shape[0]:
                if self.is_passable(grid[z + d][x]):
                    passable_neighbours[self.get_3Dstate_to_string(z + d, x)] = 1
                elif self.is_passable(grid[y+1][z+d][x]):
                    passable_neighbours[self.get_3Dstate_to_string(y+1, z+d, x)] = 1
        
        return passable_neighbours


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



    """
    Utility functions for USAR
    """

    def mark_emergency_exit_visited(self):
        """
        Utility function for USAR to avoid confusion between the
           stone_button indicating victim or emergency exit.
           TODO: This is not needed if the victims are identified by 
           other object than wool 
        """
        emergency_exit_coor = {'z': 167.5, 'x': -2192.5}
        mz = int(emergency_exit_coor['z']-self.origin_coord['z'])
        mx = int(emergency_exit_coor['x'] - self.origin_coord['x'])
        self.victims_visited[mz][mx] = 1
        self.victims_visited_sparse.add((mz, mx))


    def calc_matrix_coord(self, r_y, r_z, r_x):
        """
        Utility function: to convert 
        the relative coordinates (in the frame of player) 
        to absolute coordinates (distance wrt origin coord defined in __init__)
        """
        y_index = r_y - self.origin_coord['y']
        z_index = r_z - self.origin_coord['z']
        x_index = r_x - self.origin_coord['x']
        return y_index, z_index, x_index


    def get_2Dstate_to_string(self, z, x):
        return "S_%s_%s" % (str(z), str(x))


    def get_3Dstate_to_string(self, y, z, x):
        return "S_%s_%s_%s" % (str(y), str(z), str(x))




    # def get_player_absolute_position():
    #     """
    #     get_player_absolute_position
    #     """
    #     my = int(self.ypos - self.origin_coord['y'])
    #     mz = int(self.zpos - self.origin_coord['z'])
    #     mx = int(self.xpos - self.origin_coord['x'])

    #     absolute_position = {'y': my, 'z': mz, 'x': mx}

    #     return absolute_position



    # def get_coordinate_list_of_objects_of_interest(self, grid, item_name='wool'):
    #     close_by_items_list = []
    #     # for z in range(self.range_z//2 - self.scanning_range, self.range_z//2 + self.scanning_range + 1):
    #         # for x in range(self.range_x//2 - self.scanning_range, self.range_x//2 + self.scanning_range + 1):
    #     for z in range(0, grid.shape[0]):
    #         for x in range(0, grid.shape[1]):
    #             item = grid[z][x]
    #             # DEBUG
    #             # print(item, z, x)
    #             if item == self.object_to_index[item_name]:
    #                 print('victim', 'at', z, x)
    #                 close_by_items_list.append([z,x])
    #     return close_by_items_list
