from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import numpy as np

class NumpyMapFourRooms(MiniGridEnv):
    """
    Environment with multiple rooms and random objects.
    This environment has no specific goals or rewards.
    """

    def __init__(self, numpyFile='map000.npy'):
        self.numpyFile = numpyFile
        self.index_mapping = {
             0 : 'unseen'        ,
             1 : 'empty'         ,
             2 : 'wall'          ,
             3 : 'floor'         ,
             4 : 'door'          ,
             5 : 'key'           ,
             6 : 'ball'          ,
             7 : 'box'           ,
             8 : 'goal'          ,
             9 : 'lava'          ,
             10: 'agent'          
        }
        super().__init__(grid_size=41, max_steps=1000)


    def _gen_grid(self, width, height):
        
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Create the grid
        self.array = np.load(self.numpyFile)

        for i in range(1, self.array.shape[0]):
            for j in range(1, self.array.shape[1]):
                entity_name = index_mapping[self.array[i][j]]
                entity_index = int(self.array[i][j])

                if entity_index != 10 and entity_index != 0:  #'agent':
                    for entity_class in WorldObj.__subclasses__():
                        # the class name needs to be lowercase (not sentence case)
                        if entity_name == entity_class.__name__.casefold():
                            #     print('entity_index')
                            
                            self.put_obj(entity_class(), j, i)

                elif entity_index == 0:
                    self.put_obj(Wall(), j, i) 
        self.place_agent()
        self.place_obj(Goal())
        self.mission = 'Reach the goal'            


    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        return obs, reward, done, info


class NumpyMapMinecraftUSAR(MiniGridEnv):

    def __init__(self, numpyFile='numpy_worlds/grid.npy', agent_start_pos=[24, 25], agent_start_dir=2):
        self.numpyFile = numpyFile
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.index_mapping = {
            9: 'empty',
            8: 'agent',
            1: 'agent',
            2: 'door',
            4: 'wall',
            5: 'lava',
            6: 'key',
            7: 'goal',
            3: 'goal',
            0: 'unseen',
            10: 'box', 
            -1: 'box'
        }
        self.color_mapping = {
            9: '',
            8: '',
            1: '',
            2: 'green',
            4: 'grey',
            5: '',
            6: 'yellow',
            7: '',
            3: '',
            0: '',
            10: 'yellow', 
            -1: 'red'
        }
        self.toggletimes_mapping = {
            9: 0,
            8: 0,
            1: 0,
            2: 1,
            4: 0,
            5: 0,
            6: 1,
            7: 5,
            3: 5,
            0: 0,
            10: 3, 
            -1: 2
        }
        super().__init__(grid_size=50, max_steps=1000)


    def _gen_grid(self, width, height):
        
        # Create an empty grid
        self.grid = Grid(width, height)

        # Create the grid
        self.array = np.load(self.numpyFile)

        for mc_i in range(0, self.array.shape[0]):
            for mc_j in range(0, self.array.shape[1] ):
                mg_i , mg_j = mc_i , mc_j 
                entity_index = int(self.array[mc_i][mc_j])
                
                entity_name = self.index_mapping[entity_index]
                entity_color = self.color_mapping[entity_index]
                entity_toggletime = self.toggletimes_mapping[entity_index]

                if entity_name in ['agent', 'empty']:
                    continue
                elif entity_name == 'unseen':
                    self.put_obj(Wall(), mg_j, mg_i)
                else:
                    for entity_class in WorldObj.__subclasses__():
                        # the class name needs to be lowercase (not sentence case)
                        if entity_name == entity_class.__name__.casefold():
                            # print(entity_index, entity_name, entity_color, entity_toggletime)
                            if entity_color != '':
                                self.put_obj(entity_class(color=entity_color), mg_j, mg_i)
                            else:
                                self.put_obj(entity_class(), mg_j, mg_i)

        self.agent_pos = self.agent_start_pos
        self.grid.set(*self.agent_start_pos, None)
        self.agent_dir = self.agent_start_dir
        # self.place_obj(Goal())
        self.mission = 'Triage the victims'            

    def colorbasedreward(self, color):
        if color == 'red':
            return 10
        elif color == 'green':
            return 5
        elif color == 'blue':
            return 0
        else:
            NotImplementedError

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        if action == self.actions.forward:
            fwd_cell = self.grid.get(*self.front_pos)
            if fwd_cell != None and fwd_cell.type == 'goal':
                reward = self.colorbasedreward(fwd_cell.color)
                self.put_obj(Goal('blue'), self.front_pos[0], self.front_pos[1])
                done = False
        return obs, reward, done, info



class NumpyMapMinecraftUSARRandomVictims(NumpyMapMinecraftUSAR):
    def __init__(self, num_victims_red=10, num_victims_green=20):
        self.num_victims_red = num_victims_red
        self.num_victims_green = num_victims_green
        super().__init__()

    def _gen_grid(self, width, height):
        
        # Create an empty grid
        self.grid = Grid(width, height)

        # Create the grid
        self.array = np.load(self.numpyFile)

        for mc_i in range(0, self.array.shape[0]):
            for mc_j in range(0, self.array.shape[1] ):
                mg_i , mg_j = mc_i , mc_j 
                entity_index = int(self.array[mc_i][mc_j])
                
                entity_name = self.index_mapping[entity_index]
                entity_color = self.color_mapping[entity_index]
                entity_toggletime = self.toggletimes_mapping[entity_index]

                if entity_name in ['agent', 'empty']:
                    continue
                elif entity_name == 'unseen':
                    self.put_obj(Wall(), mg_j, mg_i)
                else:
                    for entity_class in WorldObj.__subclasses__():
                        # the class name needs to be lowercase (not sentence case)
                        if entity_name == entity_class.__name__.casefold():
                            # print(entity_index, entity_name, entity_color, entity_toggletime)
                            if entity_name == 'goal':
                                # Skip goal i.e. victim placement
                                continue 
                            elif entity_color != '':
                                self.put_obj(entity_class(color=entity_color), mg_j, mg_i)
                            else:
                                self.put_obj(entity_class(), mg_j, mg_i)

        for _ in range(self.num_victims_red):
            self.place_obj(Goal('red'))
        for _ in range(self.num_victims_green):
            self.place_obj(Goal('green'))

        self.agent_pos = self.agent_start_pos
        self.grid.set(*self.agent_start_pos, None)
        self.agent_dir = self.agent_start_dir
        # self.place_obj(Goal())
        self.mission = 'Triage the red and green victims.'     

register(
    id='MiniGrid-NumpyMapFourRooms-v0',
    entry_point='gym_minigrid.envs:NumpyMapFourRooms'
)

register(
    id='MiniGrid-NumpyMapMinecraftUSAR-v0',
    entry_point='gym_minigrid.envs:NumpyMapMinecraftUSAR'
)

register(
    id='MiniGrid-NumpyMapMinecraftUSARRandomVictims-v0',
    entry_point='gym_minigrid.envs:NumpyMapMinecraftUSARRandomVictims'
)