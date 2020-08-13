import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import gym_minigrid
from gym_minigrid.wrappers import VisdialWrapperv2 #, HumanFOVWrapper
from gym_minigrid.index_mapping import OBJECT_TO_IDX
import spacy
import numpy as np
import matplotlib.pyplot as plt
from data import train_data, test_data
ACTION_MAP = {
    0: "Turning Left",
    1: "Turning Right",
    2: "Moving forward",
    3: "Done!", 
    4: "Turning back",
}

# TODO: train supervised way with object of interest classification, later replace with general sub-goal cluster labels
# DONE: get pretrained language model
class LanguageEmbedding(nn.Module):
    def __init__(self, output_dim, embedding_dim=96, hidden_dim=128, lstm_layer=2, dropout=0.2):
        super(LanguageEmbedding, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        # self.embedding = pretrained_lm # spacy.load('en_core_web_sm') # nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding_dim = embedding_dim
        # TODO>: train spacy language embeddings
        # if static:
            # self.embedding.weight.requires_grad = False
        # self.lstm = nn.LSTM(input_size=self.embedding_dim,
        #                     hidden_size=hidden_dim,
        #                     num_layers=lstm_layer, 
        #                     dropout = dropout,
        #                     bidirectional=True)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)


    def forward(self, sentence_vectors):
        # x = torch.from_numpy(self.embedding(sents).vector)
        # x = torch.transpose(x, dim0=1, dim1=0)
        x = sentence_vectors
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        # lstm_out, (h_n, c_n) = self.lstm(x)
        # y = self.hidden2label(self.dropout(torch.cat([c_n[i,:, :] for i in range(c_n.shape[0])], dim=1)))
        return x # Softmax to identify which word it corresponds to 


# TODO: Define Conv layer size for semantic map embedding
class SemanticMapEmbedding(nn.Module):
    def __init__(self, output_dim, embedding_dim, num_objects, hidden_dim):
        super(SemanticMapEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.object_embedding = nn.Embedding(num_objects, embedding_dim)
        # self.flatout = nn.Flatten()
        
        self.conv1 = nn.Conv2d(self.embedding_dim, 64, kernel_size=3, stride=1) #, padding, dilation)
        self.conv2 = nn.Conv2d(64, 16, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=2)

        self.linear1 = nn.Linear(1152, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        # x is 50x50
        batch_size = x.size()[0]
        height = x.size()[1] 
        width = x.size()[2]
        x = x.view(batch_size, -1)
        x = self.object_embedding(x)
        
        x = x.view(batch_size, height, width, self.embedding_dim)

        x = x.permute(0, 3, 1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(batch_size, -1)
        # To check the linear dim size in case network needs to be changed
        # breakpoint()
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Policy(nn.Module):
    def __init__(self, lang_dim, map_dim,
            hidden_dim=128, num_actions=4):
        super(Policy, self).__init__()
        # self.word_embedding = LanguageEmbedding(lang_dim, lm_emb_dim)
        # self.spatial_embedding = SemanticMapEmbedding(map_dim, num_objects)

        self.fc1 = nn.Linear(lang_dim + map_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x1, x2): # human_instruction, semantic_map):
        # x1 = self.word_embedding(human_instruction)
        # x2 = self.spatial_embedding(semantic_map)
        # Add attend function!
        out = torch.cat([x1, x2], axis=1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
        
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


def is_passable_object(grid_item, impassable_objects=[2,30,9]):
    # TBD : needs updating
    # if grid_item in [9, 5, 2, 6]:  # ['air', 'fire', 'wooden_door'] 
    if grid_item in impassable_objects:
        return False
    return True


# def get_path_matrix(map_layout, index_0, index_1):
#     # agent_pos
#     path_matrix = np.zeros(map_layout.shape) #, dtype=np.int32)
#     path_matrix[index_0, index_1] = 1
#     queue = [[index_0, index_1]]

#     while len(queue):
#         coordinate = queue.pop(0)           
#         # print('coordinate', coordinate)
#         coord_z, coord_x  = coordinate
#         # print('coord_z', coord_z, 'coord_x', coord_x)

#         for diff in [-1, 1]:                
#             if is_passable_coordinate(map_layout, coord_z + diff, coord_x):
#                 # if path_matrix[coord_z + diff][coord_x] == 0:
#                 if not (path_matrix[coord_z + diff][coord_x] or 0):
#                     path_matrix[coord_z + diff][coord_x] =  path_matrix[coord_z][coord_x] + 1 # max(-1e-10, discount_factor * path_matrix[coord_z][coord_x] - time_penalty)
#                     queue.append([coord_z + diff, coord_x])

#             if is_passable_coordinate(map_layout, coord_z, coord_x + diff):   
#                 # if path_matrix[coord_z][coord_x + diff] == 0:
#                 if not (path_matrix[coord_z][coord_x + diff] or 0):
#                     path_matrix[coord_z][coord_x + diff] = path_matrix[coord_z][coord_x] + 1 # max(-1e-10, discount_factor * path_matrix[coord_z][coord_x] - time_penalty)
#                     queue.append([coord_z, coord_x + diff])

#     return path_matrix

def get_path_matrix(absolute_map, index_0, index_1, reward=100, discount_factor=0.99, time_penalty=0.01):
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
    print("Expert's choices:", [ACTION_MAP[i[0]] for i in indices])
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


def get_path_matrices_for_target(env, map_layout, target_index):
    # astar or DP (flood-fill)
    # DP (flood-fill)
    # If there is any door index then create its flood fill matrix and cache it
    # Based on all the matrices, we extract the subgoal which maximizes the value at current step of the agent.
    
    # cache path matrices!!!
    # breakpoint()

    indices = np.argwhere(map_layout == target_index)
    
    path_matrices = []
    for i in range(indices.shape[0]):
        path_matrices.append(get_path_matrix(map_layout, indices[i][0], indices[i][1]))
    
    if not len(path_matrices):
        return None

    # breakpoint()
    return path_matrices, indices


def get_index(path_matrices, env, remove_pos):

    value_at_agent_pos = np.zeros(len(path_matrices))
    for i, matrix in enumerate(path_matrices):
        if i not in remove_pos:
            value_at_agent_pos[i] = matrix[env.agent_pos[1], env.agent_pos[0]]
    
    index = np.argmax(value_at_agent_pos)
    return index

def main(args):
    num_steps = 10
    visited_list = []
    remove_pos = []
    device = torch.device("cuda" if args.use_cuda else "cpu")
    
    env = gym.make('MiniGrid-MinimapForSparky-v0')
    env = VisdialWrapperv2(env)
    obs = env.reset()

    actual_map = env.grid.encode()[:,:,0].T
    nlp = spacy.load('en_core_web_sm')
    embedding_dim = nlp('dummy text here').vector.shape[0]
    
    languageModel = LanguageEmbedding(output_dim=args.lang_emb_dim, embedding_dim=embedding_dim)
    mappingModel = SemanticMapEmbedding(output_dim=args.map_emb_dim, embedding_dim=10, 
        num_objects=len(OBJECT_TO_IDX), hidden_dim=128)
    policyModel = Policy(args.lang_emb_dim, args.map_emb_dim)

    def concat(list_of_generators):
        for generator in list_of_generators:
            yield from generator

    model_params = concat([languageModel.parameters(), mappingModel.parameters(), policyModel.parameters()])
    optimizer = torch.optim.Adam(model_params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    # path_matrices = action_path_plan(agent_pos, map_layout, target_index=2)
    
    for target_obj, human_instructions in test_data.items():
        for step in range(args.epochs):
            # path_matrices, index = action_path_plan(env,
            #             actual_map,
            #             # semantic_map,
            #             OBJECT_TO_IDX[target_obj], 
            #             visited_list)
            path_matrices, indices = get_path_matrices_for_target(env, actual_map, OBJECT_TO_IDX[target_obj])
            # indices = np.delete(indices, visited_list).reshape(-1, 2)
            # path_matrices = path_matrices.delete()
            
            for z,x in visited_list:
                remove_pos.append(np.argwhere(np.logical_and(indices[:, 0]==z, indices[:, 1]==x)))
            
            index = get_index(path_matrices, env, remove_pos)

            
            for human_instruction in human_instructions:
                print(human_instruction)
                # breakpoint()
                done = False

                for step in range(num_steps):

                    lang_vector = torch.tensor(nlp(human_instruction).vector).to(device).unsqueeze_(0)
                    lang_embeds = languageModel(lang_vector)
                    
                    belief_mask = env.observed_absolute_map 
                    semantic_map = belief_mask * actual_map
                    # Convert to torch 
                    semantic_map_tensor = torch.tensor(semantic_map).to(device).unsqueeze_(0).long() # Added until VecEnv are added in.
                    map_embeds = mappingModel(semantic_map_tensor)
                    # breakpoint()

                    # Expert action based on value estimate
                    
                    expert_action = get_solution_path(path_matrices[index], env)

                    if expert_action is None:
                        print("Can't execute the command as not observed.")
                        break 
                    elif expert_action == 3:
                        if [env.agent_pos[1], env.agent_pos[0]] not in visited_list:
                            visited_list.append([env.agent_pos[1], env.agent_pos[0]])
                        
                        for z,x in visited_list:
                            remove_pos.append(np.argwhere(np.logical_and(indices[:, 0]==z, indices[:, 1]==x)))
            
                        index = get_index(path_matrices, env, remove_pos)
                        done = True
                    expert_action = torch.tensor(expert_action).unsqueeze_(0)
                    
                    # Predicted action by the model based on the map and instruction
                    
                    optimizer.zero_grad()
                    predicted_action_vec = policyModel(lang_embeds, map_embeds)

                    loss = criterion(predicted_action_vec, expert_action)

                    loss.backward()
                    optimizer.step()

                    if args.save_model:
                        pass
                    if step % args.log_interval == 0:
                        pass

                    # print(predicted_action_vec) 
                    predicted_action = torch.argmax(predicted_action_vec, axis=1).item()
                    # breakpoint()

                    print('expert_action:', ACTION_MAP[expert_action.item()])
                    print('predicted_action:', ACTION_MAP[predicted_action])
                    print('\n')
                    # if predicted_action == 3:
                    # if expert_action == 3:
                    #     # Done!
                    #     break
                    # Update the env with expert_action
                    obs,_,_,_ = env.step(expert_action)
                    # obs,_,_,_ = env.step(predicted_action)
                    # Debug
                    t = 0.5
                    if done: 
                        response = "Done!"
                        t = 1.5
                    else: 
                        response = "Going for it~"
                        t = 0.2
                    img = env.render()
                    plt.clf()
                    plt.subplot(121)
                    plt.imshow(img)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Map layout')
                    plt.subplot(122)
                    im = plt.imshow(path_matrices[index], cmap='jet')
                    plt.colorbar(im,fraction=0.046, pad=0.04)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Value matrix')
                    plt.suptitle(f'Human: {human_instruction} \n Robot: {response}')
                    plt.draw()
                    plt.pause(t)




def test_LanguageEmbedding(args):
    nlp = spacy.load('en_core_web_sm')
    embedding_dim = nlp('dummy text here').vector.shape[0]
    output_dim = args.lang_emb_dim  
    # print(embedding_dim)
    model = LanguageEmbedding(output_dim, embedding_dim)
    batch_size = args.batch_size
    
    x = [] 
    for i, (item, sentences) in enumerate(test_data.items()):    
        for sentence in sentences:
            x.append(nlp(sentence).vector)
    
    x = torch.tensor(x)
    out = model(x)
    return out


def test_SemanticMapEmbedding(args):
    env = gym.make('MiniGrid-MinimapForSparky-v0')
    # env = HumanFOVWrapper(env)
    obs = env.reset()
    actual_map = env.grid.encode()[:,:,0]
    
    # Feeding this to semantic map embedding network
    model = SemanticMapEmbedding(output_dim=20, embedding_dim=10, num_objects=10, hidden_dim=128)
    # breakpoint()
    actual_map = torch.tensor([actual_map]*args.batch_size).long()
    out = model(actual_map)
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Navigation Policy')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    # parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
    #                     help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # parser.add_argument('--dry-run', action='store_true', default=False,
    #                     help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--lang-emb-dim', type=int, default=10,
                        help='For language embedding to be fed in Navigation Policy')
    parser.add_argument('--map-emb-dim', type=int, default=20,
                        help='For semantic map embedding to be fed in Navigation Policy')
    args = parser.parse_args()
    main(args)
    # out = test_LanguageEmbedding(args)
    # out = test_SemanticMapEmbedding(args)

    breakpoint()
    print('done')
