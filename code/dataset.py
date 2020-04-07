import torch
from torch.utils.data import Dataset

class DirectionalFrontiers(torch.utils.data):
    """
    Input : trajectory map, agent direction
    Output: subset_frontier_map
    """
    def __init__(self, root_dir, start_episode_num, max_episodes, max_count):
        self.root_dir = root_dir
        self.start_episode_num = start_episode_num
        self.max_episodes = max_episodes
        self.max_count = max_count
        self.num_directions = 4

        self.envsize = 50

        self.data = self.retrieve_all_episodes()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_frontier_matrix(self, frontier_list, mask):
        frontier_map = np.zeros((self.envsize, self.envsize)) 
        # Z = len(frontier_list)
        for idx, coord in enumerate(frontier_list):
            if mask[idx]:
            # For now all frontiers are equally likely
                frontier_map[coord[0]][coord[1]] = 1.
        return frontier_map


    def retrieve_episode_count_data(self, episode_num, counter, direction):
        """Retrieves data for a single timestep in an episode
        Input: 
            episode num : the run index in .npz
            counter : the count index .npz 

        Returns:
            trajectory_map : 2D numpy array (envsize x envsize)
            agent direction : 1D numpy array
            instruction : 1-hot vector of size 4

            subset frontier map : 2D numpy array (envsize x envsize)
        """
        episode_data = np.load(self.root_dir + '/run{}_count{}.npz'.format(episode_num, counter)) 
        # frontier_list = self.get_absolute_frontier_coordinates(episode_data['trajectory_map'])
        subset_frontier_matrix = self.get_frontier_matrix(episode_data['frontier_coordinates'], episode_data['directional_frontiers'][direction])
        # for mask in episode_data['directional_frontiers'][direction]:
        instruction = np.zeros((4))
        instruction[direction] =  1
        return episode_data['trajectory_map'], episode_data['agent_dir'], instruction, subset_frontier_matrix


    def retrieve_all_counts(self, episode_num):
        data = []
        for counter in range(self.max_count):
            data.append(self.retrieve_episode_count_data(episode_num, counter, 'south'))
            data.append(self.retrieve_episode_count_data(episode_num, counter, 'east'))
            data.append(self.retrieve_episode_count_data(episode_num, counter, 'north'))
            data.append(self.retrieve_episode_count_data(episode_num, counter, 'west'))
        return data


    def retrieve_all_episodes(self):
        data = []
        for episode in range(self.max_episodes):
            data.append(self.retrieve_all_counts(episode))
        return data
