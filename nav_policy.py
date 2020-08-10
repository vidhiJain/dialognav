import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import gym_minigrid
# from gym_minigrid.wrappers import HumanFOVWrapper
# from gym_minigrid.index_mapping import OBJECT_TO_IDX
import spacy
import numpy as np

from data import train_data, test_data


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
        self.object_embedding = nn.Embedding(num_objects, embedding_dim)
        self.flatout = nn.Flatten()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

        # self.conv1 = nn.Conv2d()
        # self.conv2 = nn.Conv2d()

    def forward(self, x):
        # x is 50x50
        x = self.object_embedding(x)
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        x = self.flatout(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Policy(nn.Module):
    def __init__(self, lang_dim, map_dim, hidden_dim=128, num_actions=4):
        self.word_embedding = LanguageEmbedding()
        self.spatial_embedding = SemanticMapEmbedding(lang_dim, len(OBJECT_TO_IDX))

        self.fc1 = nn.Linear(lang_dim + map_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, human_instruction, semantic_map):
        x1 = self.word_embedding(human_instruction)
        x2 = self.spatial_embedding(semantic_map)
        out = torch.cat([x1, x2])
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
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


def is_passable_object(grid_item, impassable_objects=[4,30,9]):
    # TBD : needs updating
    # if grid_item in [9, 5, 2, 6]:  # ['air', 'fire', 'wooden_door'] 
    if grid_item in impassable_objects:
        return False
    return True


def get_path_matrix(map_layout, index):
    # agent_pos
    path_matrix = np.zeros(map_layout.shape) #, dtype=np.int32)
    path_matrix[agent_pos[0], agent_pos[1]] = 1
    queue = [[agent_pos[0], agent_pos[1]]]

    while len(queue):
        coordinate = queue.pop(0)           
        # print('coordinate', coordinate)
        coord_z, coord_x  = coordinate
        # print('coord_z', coord_z, 'coord_x', coord_x)

        for diff in [-1, 1]:                
            if is_passable_coordinate(map_layout, coord_z + diff, coord_x):
                # if path_matrix[coord_z + diff][coord_x] == 0:
                if not (path_matrix[coord_z + diff][coord_x] or 0):
                    path_matrix[coord_z + diff][coord_x] =  path_matrix[coord_z][coord_x] + 1 # max(-1e-10, discount_factor * path_matrix[coord_z][coord_x] - time_penalty)
                    queue.append([coord_z + diff, coord_x])

            if is_passable_coordinate(map_layout, coord_z, coord_x + diff):   
                # if path_matrix[coord_z][coord_x + diff] == 0:
                if not (path_matrix[coord_z][coord_x + diff] or 0):
                    path_matrix[coord_z][coord_x + diff] = path_matrix[coord_z][coord_x] + 1 # max(-1e-10, discount_factor * path_matrix[coord_z][coord_x] - time_penalty)
                    queue.append([coord_z, coord_x + diff])

    return path_matrix


def get_solution_path(path_matrix, agent_pos):
    values = []
    for d in [-1, 1]:
        values.append(path_matrix[agent_pos[0]+d, agent_pos[1]])
        values.append(path_matrix[agent_pos[0], agent_pos[1]+d])
    index = np.argmin(np.array(values))
    return index


def action_path_plan(agent_pos, map_layout, target_index=2):
    # astar or DP (flood-fill)
    # DP (flood-fill)
    # If there is any door index then create its flood fill matrix and cache it
    # Based on all the matrices, we extract the subgoal which maximizes the value at current step of the agent.
    
    # cache
    indices = np.argwhere(map_layout == target_index)
    path_matrices = []
    for index in indices:
        path_matrices.append(get_path_matrix(map_layout, index))
    
    value_at_agent_pos = np.array(len(path_matrices))
    for matrix in path_matrices:
        value_at_agent_pos = matrix[agent_pos[0], agent_pos[1]]
    
    index = np.argmin(value_at_agent_pos)
    actions = get_solution_path(path_matrices[index], agent_pos)
    # TODO: pass
    return action


def main(args):

    device = torch.device("cuda" if args.use_cuda else "cpu")
    
    env = gym.make('MiniGrid-MinimapForSparky-v0')
    # env = HumanFOVWrapper(env)
    obs = env.reset()

    actual_map = env.grid.encode()[:,:,0]

    model = Policy()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    # path_matrices = action_path_plan(agent_pos, map_layout, target_index=2)
    for step in range(args.epochs):
        for target_obj, human_instruction in train_data.items():
            belief_mask = env.observed_absolute_map 
            semantic_map = belief_mask * actual_map
            # Convert to torch 
            semantic_map = torch.tensor(semantic_map).to(device)

            # Expert action based on value estimate
            expert_action = action_path_plan(semantic_map, OBJECT_TO_IDX[target_obj])

            # Predicted action by the model based on the map and instruction
            predicted_action = model(human_instruction, semantic_map)

            loss = criterion(predicted_action, expert_action)

            loss.backward()
            optimizer.step()

            if args.save_model:
                pass
            if step % args.log_interval == 0:
                pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Navigation Policy')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
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
    args = parser.parse_args()
    # main(args)
    nlp = spacy.load('en_core_web_sm')
    embedding_dim = nlp('hello world').vector.shape[0]
    # print(embedding_dim)
    model = LanguageEmbedding(output_dim=5, embedding_dim=embedding_dim)
    batch_size = args.batch_size
    
    x = [] 
    for i, (item, sentences) in enumerate(test_data.items()):    
        for sentence in sentences:
            x.append(nlp(sentence).vector)
    
    x = torch.tensor(x)
    out = model(x)
    
    breakpoint()
    print('done')
