import os
import numpy as np
import torch
import torch.nn as nn
import pdb
import sys


class Network(nn.Module):

    def __init__(self, input_channels=1):
        '''
        input_image_dim = 50x50x1
        '''
        super(Network, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1), #48
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2), #24
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2), #12
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=100, kernel_size=2, stride=2), #6
            nn.ReLU(),
            )

        self.combine_embeddings = nn.Sequential(
                                                nn.Linear(in_features=(6*6*100)+5, out_features=(6*6*100)),
                                                nn.BatchNorm1d(6*6*100),
                                                )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100,out_channels=64, kernel_size=2, stride=2), #12
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64,out_channels=32, kernel_size=2, stride=2), #24
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32,out_channels=16, kernel_size=2, stride=2), #48
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16,out_channels=1, kernel_size=3, stride=1), #50
            nn.Sigmoid(),
            )

    
    def forward(self, x_map, agent_dir, instr):
        '''
        x: (batch_size, channels, height, width)
            Each sample in the batch is one image
        '''
        batch_size = x_map.size()[0]    
        x_map = x_map.float()
        agent_dir = agent_dir.float()
        instr = instr.float()
        # import ipdb; ipdb.set_trace()
        encoded_map = self.encoder(x_map)
        encoded_map = encoded_map.view(batch_size, 100*6*6)
        concat_vec = torch.cat((encoded_map, instr, agent_dir),1)
        concat_vec = self.combine_embeddings(concat_vec)
        concat_vec = concat_vec.view(batch_size, 100, 6, 6)
        frontier_map = self.decoder(concat_vec)

        return frontier_map
