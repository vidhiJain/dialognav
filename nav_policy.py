import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import gym_minigrid
from gym_minigrid.wrappers import HumanFOVWrapper
from gym_minigrid.index_mapping import OBJECT_TO_IDX

from data import train_data, test_data


# TODO: get pretrained language model
class LanguageEmbedding(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=128, lstm_layer=2, dropout=0.2, embedding_dim=9):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer, 
                            dropout = dropout,
                            bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim*lstm_layer*2, embedding_dim)
    
    def forward(self, sents):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        lstm_out, (h_n, c_n) = self.lstm(x)
        y = self.hidden2label(self.dropout(torch.cat([c_n[i,:, :] for i in range(c_n.shape[0])], dim=1)))
        return y


# TODO: Define Conv layer size for semantic map embedding
class SemanticMapEmbedding(nn.Module):
    def __init__(self, num_objects, embedding_dim):
        self.object_embedding = nn.Embedding(num_objects, embedding_dim)
        self.flatout = nn.Flatten()
        self.linear = nn.Linear()
        # self.conv1 = nn.Conv2d()
        # self.conv2 = nn.Conv2d()

    def forward(self, x):
        # x is 50x50
        out = self.object_embedding(x)
        # out = F.relu(self.conv1(x))
        # out = F.relu(self.conv2(x))
        out = self.flatout(out)
        return out


class Policy(nn.Module):
    def __init__(self, lang_dim, map_dim, hidden_dim=128, num_actions=4):
        self.word_embedding = LanguageEmbedding()
        self.spatial_embedding = SemanticMapEmbedding(len(OBJECT_TO_IDX), map_dim)

        self.fc1 = nn.Linear(lang_dim + map_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, human_instruction, semantic_map):
        x1 = self.word_embedding(human_instruction)
        x2 = self.spatial_embedding(semantic_map)
        out = torch.cat([x1, x2])
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return out
        

def action_path_plan(map_layout, target_index=2):
    # astar or DP (flood-fill)
    # DP (flood-fill)
    # If there is any door index then create its flood fill matrix and cache it
    # Based on all the matrices, we extract the subgoal which maximizes the value at current step of the agent.
    indices = np.argwhere(map_layout == target_index)
    # TODO: pass
    return action


def main(args):

    device = torch.device("cuda" if args.use_cuda else "cpu")
    
    env = gym.make('MiniGrid-MinimapForSparky-v0')
    env = HumanFOVWrapper(env)
    obs = env.reset()

    actual_map = env.grid.encode()[:,:,0]

    model = Policy()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for step in range(args.epochs):
        for target_obj, human_instruction in train_data.items():
            belief_mask = env.observed_absolute_map 
            semantic_map = belief_mask * actual_map

            # Convert to torch 
            semantic_map = torch.tensor(semantic_map).to(device)


            # Expert action based on value estimate
            expert_action = action_path_plan(, OBJECT_TO_IDX[target_obj])

            # Predicted action by the model based on the map and instruction
            predicted_action = model(human_instruction, semantic_map)

            loss = criterion(predicted_action, expert_action)

            loss.backward()
            optimizer.step()

            if args.save_model:
                pass
            if step % args.log_interval == 0:
                pass


if if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Navigation Policy')
    # parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                     help='input batch size for training (default: 64)')
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
    main(args)