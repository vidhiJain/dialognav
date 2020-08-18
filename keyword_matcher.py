# import torch
import pandas as pd

import torch
from torch.utils.data import TensorDataset, Dataloader
from transformers import pipeline
from data import test_data, train_data
import nav_expert

import gym
import gym_minigrid
from gym_minigrid.wrappers import VisdialWrapperv2
from gym_minigrid.index_mapping import OBJECT_TO_IDX

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

def eval_zero_shot_classification():
    acc = 0
    correct = 0
    total = 0
    for target, sentences in train_data.items():
        for sequence in sentences:
            out = classifier(sequence, candidate_labels)
            prediction = out['labels'][0]
            print('prediction', prediction, ':', OBJECT_MAP[prediction], 'target', target)
            correct += int(OBJECT_MAP[prediction] == target)
            print('correct,', correct)
            total += 1
    print('Acc', (100*correct)/total)
    # = train_df[0]


def get_dataloaders():
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data.csv')
    inputs = torch.tensor(train_df[0].values)
    target = torch.tensor(train_df[1].values)

    train_dataset = TensorDataset(inputs, target)
    train_dataloader = Dataloader(train_dataset, batch_size=8, shuffle=True)

    # test 

    return train_dataloader


def main():

    env = gym.make('MiniGrid-MinimapForSparky-v0')
    env = VisdialWrapperv2(env)
    
    obs = env.reset()
    actual_map = env.grid.encode()[:,:,0].T

    # semantic_map = belief_mask * actual_map
    for i in range(5):
        obs, _, _, _ = env.step(0)
        belief_mask = env.observed_absolute_map 
        semantic_map = np.where(belief_mask, actual_map, -1)

        target_obj = 'goal'
        target_index = OBJECT_TO_IDX[target_obj]
        indices = np.argwhere(semantic_map == target_index)
        breakpoint()

        # Extract frontier coordinates if cannot reach the target.
        if indices.shape[0] == 0:  
            # target_index = 'frontier'
            frontier_list = get_frontier_list(semantic_map)
            indices = np.array(frontier_list)
            
        # frontier_map = get_frontier_map(semantic_map)
        # import matplotlib.pyplot as plt
        # plt.matshow(frontier_map)
        # plt.show()
        
        # TODO: cache them!
        path_matrices = get_path_matrices_for_target(env, semantic_map, indices)