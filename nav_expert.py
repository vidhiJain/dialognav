import numpy as np

ACTION_MAP = {
    0: "Turning Left",
    1: "Turning Right",
    2: "Moving forward",
    3: "Done!", 
    4: "Turning back",
}
candidate_labels = ["door", "room", 
                    "injured", "victim", # "person", 
                    "light switch", "lever", "switch", "electric switch", 
                    "fire",
                    "shovel", "pickaxe", "axe", "water"]

OBJECT_MAP = {
    'door': 'door',
    'room': 'door', 
    
    'victim':  'goal',
    'injured': 'goal',
    'person': 'goal',

    'light switch': 'key',
    'lever': 'key',
    'switch': 'key',
    'electric switch': 'key',

    'fire': 'lava', 
    
    'shovel': 'ball',
    'pick axe': 'ball',
    'axe': 'ball',
    'water': 'ball'

}

CODE_TO_COMMON_MAP = {
    'goal': ['victim', 'injured', 'casualities', 'who may need help', 'people', 'affected'], 
    'key': ['switch', 'electric switch', 'lever', 'light switch'],
    'door': ['door'],
    'lava': ['fire', 'hazard'],
    'ball': ['water', 'shovel', 'pickaxe', 'axe']
}


# def is_passable_coordinate(map_layout, coord_z, coord_x):
def is_passable_coordinate(grid, z, x):
    # # Hole on the floor; can't pass
    # if self.is_passable_object(grid[0, z, x]):
    #     return False

    # same level or above is passable; then can pass
    if z >= 0 and x >= 0 and z < grid.shape[0] and x < grid.shape[1]:
        if is_passable_object(grid[z, x]):
            return True
    return False


def is_passable_object(grid_item, impassable_objects=[2,30,9,-1]):
    # TBD : needs updating
    # if grid_item in [9, 5, 2, 6]:  # ['air', 'fire', 'wooden_door'] 
    if grid_item in impassable_objects:
        return False
    return True


def get_path_matrix(absolute_map, index_0, index_1, reward=2, discount_factor=0.99, time_penalty=0.01):
    # import ipdb; ipdb.set_trace()
    path_matrix = np.zeros(absolute_map.shape) #, dtype=np.int32)
    path_matrix[index_0, index_1] = reward
    queue = [[index_0, index_1]]

    while len(queue):
        coordinate = queue.pop(0)           
        # print('coordinate', coordinate)
        coord_z, coord_x  = coordinate
        # print('coord_z', coord_z, 'coord_x', coord_x)

        for diff in [-1, 1]:                
            if is_passable_coordinate(absolute_map, coord_z + diff, coord_x):
                # if path_matrix[coord_z + diff][coord_x] == 0:
                if not (path_matrix[coord_z + diff][coord_x] or 0):
                    path_matrix[coord_z + diff][coord_x] =  max(-1e-10, discount_factor * path_matrix[coord_z][coord_x] - time_penalty)
                    queue.append([coord_z + diff, coord_x])

            if is_passable_coordinate(absolute_map, coord_z, coord_x + diff):   
                # if path_matrix[coord_z][coord_x + diff] == 0:
                if not (path_matrix[coord_z][coord_x + diff] or 0):
                    path_matrix[coord_z][coord_x + diff] =  max(-1e-10, discount_factor * path_matrix[coord_z][coord_x] - time_penalty)
                    queue.append([coord_z, coord_x + diff])

    return path_matrix


def get_value(path_matrix, coordinates):
    return path_matrix[coordinates[1], coordinates[0]]


def get_solution_path(path_matrix, env):
    values = []
    neighbour_value = np.array([get_value(path_matrix, env.left_pos), get_value(path_matrix, env.right_pos), 
        get_value(path_matrix, env.front_pos), get_value(path_matrix, env.agent_pos),
        get_value(path_matrix, env.back_pos)])
    indices = np.argwhere(neighbour_value == np.amax(neighbour_value))
    # print("Expert's choices:", [ACTION_MAP[i[0]] for i in indices])
    # breakpoint()
    
    if 4 in indices:
        return 0 # Arbitrarily turn left?!
    if 2 in indices:
        return 2
    
    return np.random.choice(indices.reshape(-1)) 
    # for d in [-1, 1]:
    #     values.append(path_matrix[agent_pos[0]+d, agent_pos[1]])
    #     values.append(path_matrix[agent_pos[0], agent_pos[1]+d])
    # index = np.argmin(np.array(values))
    # if agent_dir == 
    # return index



def get_path_matrices_for_target(env, map_layout, indices):
    # astar or DP (flood-fill)
    # DP (flood-fill)
    # If there is any door index then create its flood fill matrix and cache it
    # Based on all the matrices, we extract the subgoal which maximizes the value at current step of the agent.
    
    # cache path matrices!!!
    # breakpoint()


    path_matrices = []
    for i in range(indices.shape[0]):
        path_matrices.append(get_path_matrix(map_layout, indices[i][0], indices[i][1]))
    
    if not len(path_matrices):
        return None

    # breakpoint()
    return path_matrices


def get_index(path_matrices, env, actual_pos, remove_pos):
    # breakpoint()
    value_at_agent_pos = np.zeros(len(path_matrices))
    for i, matrix in enumerate(path_matrices):
        if i in actual_pos and i not in remove_pos:
            value_at_agent_pos[i] = matrix[env.agent_pos[1], env.agent_pos[0]]
    
    index = np.argmax(value_at_agent_pos)
    return index


def get_frontier_list(map_layout):
    frontier_list = []
    # neighbour_avg_map = np.where(map_layout, get_neighbour_avg(), map_layout)
    for i in range(1, map_layout.shape[0]-1):
        for j in range(1, map_layout.shape[1]-1):
            if is_passable_coordinate(map_layout, i, j):
                
                for d in [-1,1]:
                    if map_layout[i][j+d] == -1:
                        frontier_list.append([i, j])
                        break
                    if map_layout[i+d][j] == -1:
                        frontier_list.append([i, j])
                        break
                # if avg_neighbour <= 0:
                    
    return frontier_list


def get_frontier_map(indices, semantic_map):
    for [x,y] in indices: 
        semantic_map[x][y] = OBJECT_TO_IDX['frontier']
    return semantic_map


def get_response(out, flag_done, flag_max_steps, flag_frontier, clarity_threshold=0.5): 
    name_list = CODE_TO_COMMON_MAP[OBJECT_MAP[out['labels'][0]]]
    if out['labels'][0] in name_list:
        target = out['labels'][0]
    else:
        target = name_list[0]

    if flag_done:
        if flag_max_steps:
            return "I tried, but could not find any " + target
        elif flag_frontier:
            return "I reached an unseen area, but haven't found " + target + ". Should I continue to explore?"
        else:    
            return "Done! I reached the " + target
    if flag_frontier:
        return "I am searching for " + target + " in unexplored areas."
    if out['scores'][0] > clarity_threshold:
        response = "I am going for the nearest " + target
    else: 
        if out['labels'][0] in name_list and out['labels'][1] in name_list:
            response = "I am going for it."
        else:
            response = "I am going for " + out['labels'][0] + ", but should I navigate to " + out['labels'][1]
    return response

if __name__ == "__main__":
    import gym
    import gym_minigrid
    from gym_minigrid.wrappers import VisdialWrapperv2
    from gym_minigrid.index_mapping import OBJECT_TO_IDX
    from data import test_data, train_data, demo_data
    from transformers import pipeline
    import matplotlib.pyplot as plt
    max_steps = 10

    classifier = pipeline("zero-shot-classification")

    

    # env = gym.make('MiniGrid-MinimapForSparky-v0')
    env = gym.make('MiniGrid-MinimapForFalcon-v0')
    env = VisdialWrapperv2(env)
    
    obs = env.reset()
    
    # # To make it east facing :P
    # for _ in range(2):
    #     env.step(0)
    
    # Take some random actions for fun
    for _ in range(10):
        for i in [0]:
            env.step(i)

    actual_map = env.grid.encode()[:,:,0].T

    visited_list = []
    remove_pos = []
    
    # semantic_map = belief_mask * actual_map
    for target, sentences in demo_data.items():
        
        target_index = OBJECT_TO_IDX[target]
        gt_indices = np.argwhere(actual_map == target_index)
        path_matrices = get_path_matrices_for_target(env, actual_map, gt_indices)

        for sequence in sentences:
            out = classifier(sequence, candidate_labels)
            prediction = out['labels'][0]
            # print('prediction', prediction, ':', OBJECT_MAP[prediction], 'target', target)
            # correct += int(OBJECT_MAP[prediction] == target)
            # print('correct,', correct)
            # total += 1        
            target_obj = OBJECT_MAP[prediction]
            belief_mask = env.observed_absolute_map 
            semantic_map = np.where(belief_mask, actual_map, -1)

            
            
            i = 0
            flag_done = False
            flag_max_steps = False
            # target_obj = 'goal'
            while not flag_done and i < max_steps:

                # obs, rew, done, info = env.step(expert_action)
                indices = np.argwhere(semantic_map == target_index)
                flag_frontiers = False
                # breakpoint()

                # Extract frontier coordinates if cannot reach the target.
                if indices.shape[0] == 0:  
                    # target_index = 'frontier'
                    frontier_list = get_frontier_list(semantic_map)
                    indices = np.array(frontier_list)
            
                    frontier_path_matrices = get_path_matrices_for_target(env, semantic_map, indices)
                    # frontier_path_matrices are disposable since they may change with each step
                    actual_pos = [x for x in range(len(frontier_path_matrices))]
                    index = get_index(frontier_path_matrices, env, actual_pos, [])
                    expert_action = get_solution_path(frontier_path_matrices[index], env)
                    flag_frontiers = True

                else:
                    actual_pos = []

                    for z,x in indices:
                        idx_for_path_matrices = np.where(np.logical_and(gt_indices[:,0] == z, gt_indices[:,1] == x))
                        # # breakpoint()
                        # if idx_for_path_matrices[0].shape[0]:
                        actual_pos.append(idx_for_path_matrices[0][0])

                    index = get_index(path_matrices, env, actual_pos, remove_pos)
                    expert_action = get_solution_path(path_matrices[index], env)

                if expert_action is None:
                    print("Can't execute the command as not observed.")
                    break 
                elif expert_action == 3:
                    # if [env.agent_pos[1], env.agent_pos[0]] not in visited_list:
                        # visited_list.append([env.agent_pos[1], env.agent_pos[0]])
                    
                    # for z,x in visited_list:
                        # remove_pos.append(np.argwhere(np.logical_and(indices[:, 0]==z, indices[:, 1]==x)))
                    remove_pos.append(index)
                    flag_done = True
                    
                    # breakpoint()
                if i == max_steps-1:
                    flag_max_steps = True    

                obs, rew, done, info = env.step(expert_action)
                
                # breakpoint()
                response = get_response(out, flag_done, flag_max_steps, flag_frontiers)

                plt.clf()
                plt.figure(figsize=(12,6), dpi=200)
                img = env.render()
                plt.subplot(121)
                plt.imshow(img)
                plt.xticks([])
                plt.yticks([])
                plt.title('Top down view')
                
                belief_mask = env.observed_absolute_map 
                if flag_frontiers:
                    visible_path_matrix = np.where(belief_mask, frontier_path_matrices[index], -1)
                else:
                    visible_path_matrix = np.where(belief_mask, path_matrices[index], -1)
                target = f'frontier at {frontier_list[index]}' if flag_frontiers else target
                plt.subplot(122)
                plt.imshow(visible_path_matrix)
                plt.title(f'Path matrix for {target}')
                # plt.title("Human: " + sequence + "\n" + "Robot: " + response)
                plt.xticks([])
                plt.yticks([])
                plt.suptitle(f"Human: {sequence} \n Robot: {response}")
                plt.draw()
                plt.pause(0.5)

                i += 1

            ## To show the frontier map!
            # frontier_map = get_frontier_map(semantic_map)
            # import matplotlib.pyplot as plt
            # plt.matshow(frontier_map)
            # plt.show()
            
            # TODO: cache them!
    breakpoint()
    print('done')