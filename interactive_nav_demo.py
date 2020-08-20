from nav_expert import *
from transformers import pipeline
import matplotlib.pyplot as plt
import gym
import gym_minigrid
from gym_minigrid.wrappers import VisdialWrapperv2
from gym_minigrid.index_mapping import OBJECT_TO_IDX
max_steps = 10

classifier = pipeline("zero-shot-classification")

candidate_labels = ["door", "room", 
                    "injured", "victim", # "person", 
                    "light switch", "lever", "switch", "electric switch", 
                    "fire"]

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
}

CODE_TO_COMMON_MAP = {
    'goal': ['victim', 'injured', 'casualities', 'who may need help', 'people', 'affected'], 
    'key': ['switch', 'electric switch', 'lever', 'light switch'],
    'door': ['door'],
    'lava': ['fire', 'hazard'],
}

ACTION_MAP = {
    0: "Turning Left",
    1: "Turning Right",
    2: "Moving forward",
    3: "Done!", 
    4: "Turning back",
}


# env = gym.make('MiniGrid-MinimapForSparky-v0')


 

def movement(env, target_index, out, remove_pos):
    actual_map = env.grid.encode()[:,:,0].T
    belief_mask = env.observed_absolute_map 
    semantic_map = np.where(belief_mask, actual_map, -1)
    i = 0
    flag_done = False
    flag_max_steps = False
    # target_obj = 'goal'
    # while not flag_done and i < max_steps:

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
        path_matrices = get_path_matrices_for_target(env, semantic_map, indices)
        # for z,x in indices:
        #     idx_for_path_matrices = np.where(np.logical_and(gt_indices[:,0] == z, gt_indices[:,1] == x))
        #     # # breakpoint()
        #     # if idx_for_path_matrices[0].shape[0]:
        #     actual_pos.append(idx_for_path_matrices[0][0])
        actual_pos = [x for x in range(len(path_matrices))]
        index = get_index(path_matrices, env, actual_pos, remove_pos)
        expert_action = get_solution_path(path_matrices[index], env)

    if expert_action is None:
        print("Can't execute the command as not observed.")
        return 

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

    belief_mask = env.observed_absolute_map 

    if flag_frontiers:
        visible_path_matrix = np.where(belief_mask, frontier_path_matrices[index], -1.)
    else:
        visible_path_matrix = np.where(belief_mask, path_matrices[index], -1.)
    # breakpoint()
    
    target = out['labels'][0]
    target = f'frontier at {frontier_list[index]}' if flag_frontiers else target
    
    return visible_path_matrix, target, response, remove_pos, flag_done

def plot(env, visible_path_matrix, target, sentence, response):
    plt.clf()
    plt.figure(figsize=(10,4), dpi=100)
    img = env.render()
    plt.subplot(121)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.title('Top down view')
    
    
    plt.subplot(122)
    plt.imshow(visible_path_matrix) #, cmap='jet')
    plt.title(f'Path matrix for {target}')
    # plt.title("Human: " + sequence + "\n" + "Robot: " + response)
    plt.xticks([])
    plt.yticks([])
    plt.suptitle(f"Human: {sentence} \n Robot: {response}")
    plt.draw()
    plt.pause(0.5)
    # plt.close()
    

def main():
    env = gym.make('MiniGrid-MinimapForFalcon-v0')
    env = VisdialWrapperv2(env)

    obs = env.reset()
    remove_pos = []
    flag_done = False
    max_steps = 10
    
    while True:
        print('>> ')
        sentence = input()
        print('Human: ', sentence)
        try:
            out = classifier(sentence, candidate_labels)
        except:
            print("Didn't understand! Tell me where to go next.")
            continue
        # print('Robot understood :', out)
        # print(out)
        
        prediction = out['labels'][0]
        target_obj = OBJECT_MAP[prediction]
        target_index = OBJECT_TO_IDX[target_obj]
        
        i = 0
        flag_done = False
        # breakpoint()
        while i < max_steps and not flag_done:
            visible_path_matrix, target, response, remove_pos, flag_done = movement(env, target_index, out, remove_pos)
            print('Robot: ', response)
            plot(env, visible_path_matrix, target, sentence, response)
            print()
            i += 1

if __name__ == "__main__":
    main()