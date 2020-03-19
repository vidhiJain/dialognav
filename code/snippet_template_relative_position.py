import numpy as np
# 0 is unseen
# 1 is air
# 2 is fire
# 3 is victim
# 8 is the player

grid = np.array([
	[1, 1, 2, 1, 3, 1, 0],
	[1, 1, 2, 1, 3, 1, 0],
	[0, 1, 2, 1, 1, 0, 0],
	[0, 0, 0, 8, 0, 0, 0],
	[0, 0, 0, 1, 0, 0, 0],
	[0, 0, 0, 1, 0, 0, 0],
])



class Templates:
	def __init__(self, grid):
		self.grid = grid

		self.agent_position = np.array([grid.shape[0]//2, grid.shape[1]//2])
		self.index_to_object = {
			 9:'air',
			 8:'player',
			 2:'door',
			 3:'victim',
			 4:'wall',
			 5:'fire',
			 6:'lever',
			 7:'button',
			 10:'gravel'
        }

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
        self.non_opaque_list = [9, 8, 1, 2, 5, 6, 7]
        
	# def get_relative_position_statement(self, grid, object_id_1, object_id_2):
	# 	"""
	# 	Returns relative position of the two objects of interest with respect to agent
	# 	Obj1 is closer than Obj2?
	# 	Assuming obj1 and obj2 only occur once in the FOV.
	# 		 Taking their first instance from origin?
	# 	"""
	# 	n =len(grid.shape)
	# 	object_loc_1 = get_all_coordinates(grid, object_id_1)
	# 	object_loc_2 = get_all_coordinates(grid, object_id_2)
	# 	difference = np.zeros(n)
	# 	for i in range(n):
	# 		difference[i] = object_loc_1[0][i] - object_loc_2[0][i] - agent_position[i]
	# 	# z axis is closer or far
	# 	if difference[0] > 0:
	# 		statement = f'{object_id1} is closer than {object_id2}.'
	# 	elif difference[0] < 0: 
	# 		statement = f'{object_id2} is closer than {object_id1}.'
	# 	else:
	# 		statement = f'{object_id2} and {object_id1} are equally far from me.'
	# 	# x axis is left or right

	# 	return statement


	def find_object_of_interest(self, grid, object_id, line_of_sight_dist):
		"""
		Finds the relative location of the object of interest 
		Assuming only 1
		"""
		location = get_all_coordinates(grid, object_id)
		if len(location):
			if line_of_sight_dist < self.closeness_threshold :
				return f'{self.index_to_object.get(object_id)} is close to me.'

			dist = dist_func(location[-1], self.agent_position)
			if dist < line_of_sight_dist//2:
				return f'{self.index_to_object.get(object_id)} is close to me.'
			else: 
				return f'{self.index_to_object.get(object_id)} is far from me.'


	def get_aggregated_position(self, grid, object_id):
		"""
		Combines the objects spread over the map to form a concise sentence
		"""
		location = get_all_coordinates(grid, object_id)
		location = []
		z,x  = grid.shape

		for i in range(z):
			for j in range(x):
				if grid[z][x] == object_id:
					if 
					location.append([z, x])
		return location



	
# Utils (possibly)

def get_frontier_of_object_mass(grid, coordinates_list object_id):
	frontier = []
	for coord in coordinates_list:
		for d in [-1, 1]:
			if grid[coord[0] + d][coord[1]] != object_id:
				frontier.append([coord[0] + d][coord[1]])
			if grid[coord[0]][coord[1] + d] != object_id:
				frontier.append([coord[0]][coord[1] + d])
	return frontier


def get_all_coordinates(grid, object_id):
	"""
	Aggregates the list of the locations where object is.

	TODO? 
	1/ Convert it to sparse matrix Or parallelize this computation
	2/ Get 3D coordinates?
	"""

	location = []
	z,x  = grid.shape

	for i in range(z):
		for j in range(x):
			if grid[z][x] == object_id:
				location.append([z, x])
	return location
	# return np.array(location)


def dist_func(a, b, mode='l1'):
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

