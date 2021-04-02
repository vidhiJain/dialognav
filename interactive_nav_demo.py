from nav_expert import *
from transformers import pipeline
import matplotlib.pyplot as plt
import gym
import gym_minigrid
from gym_minigrid.wrappers import VisdialWrapperv2
from gym_minigrid.index_mapping import OBJECT_TO_IDX
from gym_minigrid.window import Window

import pandas as pd
import argparse
from termcolor import colored
from candidates import *

import pandas as pd
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-MinimapForSparky-v0'  #'MiniGrid-GoToObject-10x10-N3-v0' #'MiniGrid-GoToObject-8x8-N2-v0' ## 'MiniGrid-MinimapForSparky-v0' #'MiniGrid-GoToDoor-8x8-v0'
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=8
)
parser.add_argument(
    "--max_steps",
    type=int,
    help="number of steps that robot takes independently",
    default=100
)
args = parser.parse_args()
max_steps = args.max_steps

df = pd.DataFrame(columns=['timestep', 'msg_received', 
        'object', 'color', 'next-prev', 'close-far', 'reference'])


def key_handler(event):
    # print('pressed', event.key)
    global df

    if event.key == 'escape':
        df.to_csv('dialog.csv', index=False)
        window.close()
        exit(0)
        return

    # if event.key == 'backspace':
    #     reset()
    #     return

    # if event.key == 'left':
    #     step(env.actions.left)
    #     return
    # if event.key == 'right':
    #     step(env.actions.right)
    #     return
    # if event.key == 'up':
    #     step(env.actions.forward)
    #     return

    # # Spacebar
    # if event.key == ' ':
    #     step(env.actions.toggle)
    #     return
    # if event.key == 'pageup':
    #     step(env.actions.pickup)
    #     return
    # if event.key == 'pagedown':
    #     step(env.actions.drop)
    #     return

    if event.key == 'enter':
        step(env.actions.done)
        return

def onclick(event):
    global df
    # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #   ('double' if event.dblclick else 'single', event.button,
        #    event.x, event.y, event.xdata, event.ydata))
    coord = round(event.xdata / args.tile_size), round(event.ydata / args.tile_size)
    if event.dblclick:
        response = f'<< Robot: Do you want me to go there at ({coord})'
        print(response)
        with open('interactive_performance.log', 'a') as log:
            log.write(response)
        df = df.append({'time': str(datetime.now()), 'Agent': 'Robot', 'Message': response}, ignore_index=True)
    
    # print(coord)



env = gym.make(args.env)
env = VisdialWrapperv2(env, tile_size=args.tile_size)
window = Window('gym_minigrid for visdial')
window.reg_key_handler(key_handler)
window.reg_click_event(onclick)
flag_done = False

df = pd.DataFrame(columns=['time', 'Agent', 'Message'])

classifier = pipeline("zero-shot-classification")


# import logging
# logging.basicConfig(filename='interactive_performance.log',level=logging.INFO)

    
# env = gym.make('MiniGrid-MinimapForSparky-v0')


def movement(env, target_index, out, remove_pos):
    global df
    actual_map = env.grid.encode()[:,:,0].T
    belief_mask = env.observed_absolute_map 
    semantic_map = np.where(belief_mask, actual_map, -1)
    i = 0
    global flag_done
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


# def plot_2(env, visible_path_matrix, target, sentence, response):
#     plt.clf()
#     plt.figure(figsize=(10,4), dpi=100)
#     img = env.render()
#     plt.subplot(121)
#     plt.imshow(img)
#     plt.xticks([])
#     plt.yticks([])
#     plt.title('Top down view')
    
#     plt.subplot(122)
#     plt.imshow(visible_path_matrix) #, cmap='jet')
#     plt.title(f'Path matrix for {target}')
#     # plt.title("Human: " + sequence + "\n" + "Robot: " + response)
#     plt.xticks([])
#     plt.yticks([])
#     plt.suptitle(f"Human: {sentence} \n Robot: {response}")
#     plt.draw()
#     plt.pause(0.5)
#     # plt.close()
    

# def plot(env, target, sentence, response):
#     plt.clf()
#     plt.figure(figsize=(10,4), dpi=100)
#     img = env.render()
#     plt.imshow(img)
#     plt.xticks([])
#     plt.yticks([])
#     # plt.title('Top down view')
#     plt.title(f"Human: {sentence} \n Robot: {response}")
#     plt.draw()
#     plt.pause(0.5)


def step(action):
    global df
    global flag_done
    obs, reward, done, info = env.step(action)
    # breakpoint()
    # print(obs.keys())
    sentence = '>> Human: (pressed enter)'
    response = '<< Robot: I received interrupt signal. Tell me what to go for?'
    # print()
    window.set_caption(f"Human: {sentence} \n Robot: {response}")
    df = df.append({'time': str(datetime.now()), 'Agent': 'Human', 'Message': '(pressed enter)'}, ignore_index=True)
    df = df.append({'time': str(datetime.now()), 'Agent': 'Robot', 'Message': 'I received interrupt signal. Tell me what to go for?'}, ignore_index=True)

    # print(done)

    flag_done = True
    # if done:
    #     print('done!')
    #     # reset()
        
    # else:
    # redraw()
    # # print('yaw', env.yaw)


def reset():
    # if args.seed != -1:
    #     env.seed(args.seed)
    obs = env.reset()
    # print(obs['image'].shape)

    # if hasattr(env, 'mission'):
    #     print('Mission: %s' % env.mission)
    #     window.set_caption(env.mission)
    redraw(obs)


def redraw():
    img = env.render('rgb_array') #, tile_size=args.tile_size)
    window.show_img(img)


def main():
    # env = gym.make('MiniGrid-MinimapForSparky-v0')

    # env = gym_minigrid.envs.MinimapForFalcon(agent_pos=(50, 30))
    global df
    global flag_done

    obs = env.reset()
    remove_pos = []
    
    # max_steps = 10

    # Take some random actions for fun
    for _ in range(5):
        for i in [0, 2]:
            env.step(i)
    img = env.render()
    window.show_img(img)

    while True:
        print('\nConversation Logs:\n', df.tail())

        flag_done = False
        print('>> ')
        sentence = input()
        if sentence == 'end':
            break
        print(colored(f'>> Human: {sentence}', 'red'))
        try:
            out = classifier(sentence, candidate_objects)
        except:
            print("<< Robot: Didn't understand! Tell me where to go next.")
            window.set_caption(f"Human: {sentence} \n Robot: {response}")
            continue

        # DEBUG
        # print('Robot understood :', out)
        # print(out)

        
        # logging.info(sentence)
        with open('interactive_performance.log', 'a') as log:
            log.write(f'>> Human: {sentence}')
        df = df.append({'time': str(datetime.now()), 'Agent': 'Human', 'Message': sentence}, ignore_index=True)

        


        prediction = out['labels'][0]
        target_obj = OBJECT_MAP[prediction]
        target_index = OBJECT_TO_IDX[target_obj]
        
        i = 0
        prev_response = ''
        # breakpoint()
        while i < max_steps and not flag_done:
            visible_path_matrix, target, response, remove_pos, flag_done = movement(env, target_index, out, remove_pos)
            
            redraw()
            if prev_response != response:
                print('<< Robot: ', response)
                with open('interactive_performance.log', 'a') as log:
                    log.write(f'<< Robot: {response}')
                prev_response = response
                df = df.append({'time': str(datetime.now()), 'Agent': 'Robot', 'Message': response}, ignore_index=True)
                # plot(env, target, sentence, response)
            
                window.set_caption(f"Human: {sentence} \n Robot: {response}")
                
            # print()
            i += 1


if __name__ == "__main__":
    main()
    
    breakpoint()
    print('done')