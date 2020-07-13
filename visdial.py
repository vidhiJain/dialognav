#!/usr/bin/env python3
import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

from gym_minigrid.index_mapping import OBJECT_TO_IDX
from gym_minigrid.minigrid import *

def redraw(img):
    # if not args.agent_view:
    img = env.render('rgb_array') #, tile_size=args.tile_size)

    window.show_img(img)

def reset():
    # if args.seed != -1:
        # env.seed(args.seed)

    obs = env.reset()
    # print(obs['image'].shape)

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

def step(action):
    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))
    print('agent_pos', env.agent_pos)
    if done:
        print('done!')
        reset()
        # env.put_obj(Goal('blue'), env.agent_pos[0], env.agent_pos[1])
    else:
        redraw(obs)


def process_question_1(env):
    # Return compact encoding of objects in FoV for the questions:
    #     [
    #         "What do you see?", 
    #         "What is in front of you?",
    #     ]

    obj = {'door': [], 'victim': [], 'lever': []}
    current_fov = env.grid.encode()[:,:,0].T * env.visible_grid  # .reshape(-1)
    max_dist = -1
    for i in range(current_fov.shape[0]):
        for j in range(current_fov.shape[1]):
            if not current_fov[i, j] in [0, 1]:  # [unseen, visible]
                if not (i == env.agent_pos[1] and env.agent_pos[0] == j):
                    # theta = np.arctan((env.agent_pos[0] - j)/(i - env.agent_pos[1])) * 180 / np.pi
                    theta_in_rad = np.arctan2((env.agent_pos[0] - j), (i - env.agent_pos[1]))
                    theta_in_deg = (theta_in_rad * 180 / np.pi) % 360
                    
                    angle = (theta_in_deg - env.yaw) 
                    if theta_in_deg < 90 and env.yaw > 270:
                        angle = theta_in_deg + (360 - env.yaw)
                    elif theta_in_deg > 270 and env.yaw < 90:
                        angle = - (360 - theta_in_deg) - env.yaw

                    dist = (i - env.agent_pos[1])**2 + (env.agent_pos[0] - j)**2

                    if max_dist < dist:
                        max_dist = dist

                    if current_fov[i, j] == OBJECT_TO_IDX['door']:
                        obj['door'].append([dist, angle])
                    elif current_fov[i, j] == OBJECT_TO_IDX['goal']:
                        obj['victim'].append([dist, angle])
                    elif current_fov[i, j] == OBJECT_TO_IDX['key']:
                        obj['lever'].append([dist, angle])
                    else: 
                        continue

    return obj, max_dist        


def get_rel_dist(dist, threshold):
    if dist < threshold:
        return "close"
    else:
        return "far off"

def get_rel_angle(angle, threshold):
    if abs(angle) < threshold:
        return "in front of me."
    elif angle > threshold:
        return "to my right."
    elif angle < -threshold:
        return "to my left."
    return ""


def generate_response_1(obj, max_dist, items=['door', 'victim', 'lever'], more_info=True): 
        
    for item in items:
        if len(obj[item]) == 0:
            print("\nThere are no {}s".format(item))
        if len(obj[item]) == 1:
            print("\nThere is a {} {} {}".format(
                item,
                get_rel_dist(obj[item][0][0], threshold=max_dist*0.5),
                get_rel_angle(obj[item][0][1], threshold=15)
            ))
        elif len(obj[item]) > 1:
            print("\nThere are {}s.".format(item))
            
            for i, pos in enumerate(obj[item]):
                print("{}. {} {} ".format(
                    i,
                    get_rel_dist(pos[0], threshold=max_dist*0.5),
                    get_rel_angle(pos[1], threshold=15)
                ))
    if more_info:
        print("More info:", obj)
    
# def process_question_2(env):
#     env.observed_absolute_map


# def generate_response_2(ans):


def synonymous_questions(item):
    return [
        "Do you see any {}?".format(item),
        "Can you find any {}?".format(item),
        "Are any {} visible?".format(item),
        "Check for {}".format(item),
        "Check for any {}".format(item),
        "Look for {}".format(item), 
        "Look for any {}".format(item),    
    ]

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return

    if event.key == '1':
        synonymous_questions_1 = [
            "What do you see?", 
            "What is in front of you?",
            "What's visible?",
            "What can you see in front of you?",
            "Are you seeing anything interesting?",
        ]
        ind = np.random.randint(len(synonymous_questions_1))
        print('\n\n')
        print(synonymous_questions_1[ind])
        # pause the movement for question
        obj, max_dist = process_question_1(env)
        response = generate_response_1(obj, max_dist)
        print(response)

    # if event.key == '2':
    #     synonymous_questions = [
    #         "Have you been here before?",
    #         "Have you checked this room before?",
    #         "Have you seen this before?",
    #         "Did you enter this room again?",
    #         "Haven't you checked this room already?",
    #     ]
    #     ind = np.random.randint(len(synonymous_questions))
    #     print('\n\n')
    #     print(synonymous_questions[ind])
    #     ans = process_question_2(env)
    #     response = generate_response_2(ans)
    #     print(response)

    if event.key == '3':
        # 'door'
        qlist = synonymous_questions('door')
        ind = np.random.randint(len(qlist))
        print('\n\n')
        print(qlist[ind])
        # pause the movement for question
        obj, max_dist = process_question_1(env)
        response = generate_response_1(obj, max_dist, items=['door'])
        print(response)
    
    if event.key == '4':
        # 'victim'
        qlist = synonymous_questions('victim')
        ind = np.random.randint(len(qlist))
        print('\n\n')
        print(qlist[ind])
        # pause the movement for question
        obj, max_dist = process_question_1(env)
        response = generate_response_1(obj, max_dist, items=['victim'])
        print(response)

    if event.key == '5':
        # 'lever'
        qlist = synonymous_questions('lever')
        ind = np.random.randint(len(qlist))
        print('\n\n')
        print(qlist[ind])
        # pause the movement for question
        obj, max_dist = process_question_1(env)
        response = generate_response_1(obj, max_dist, items=['lever'])
        print(response)

    # if event.key == '6':
    #     # frontiers
        

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-MinimapForSparky-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=8
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)

# args = parser.parse_args()

# env = gym.make(args.env)
env = gym.make('MiniGrid-MinimapForSparky-v0')
env = VisdialWrapper(env, 8, agent_pos=(24,25))

# env = FullyObsWrapper(env)
# env = gym.wrappers.Monitor(env, "recording")
#env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True,force=True)
# if args.agent_view:
    #env = RGBImgPartialObsWrapper(env)
    # env = ImgObsWrapper(env)

window = Window('gym_minigrid - ' + 'minigrid-minimapforsparky-v0')
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
