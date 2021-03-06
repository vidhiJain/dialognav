import numpy as np
import sys
import os
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from human_to_robot_network import Network
from code.dataset import DirectionalFrontiers


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, 
        help='learning rate')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, 
        help='batch_size')
    parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=60, 
        help='number of epochs')
    parser.add_argument('--data-dir', dest='data_dir', type=str, default='./train_data', 
        help='path to data files')
    parser.add_argument('--log-interval', dest='log_interval', type=int, default=10, 
        help='Loss summary interval')
    return parser.parse_args()


def plot_props(data, prop_name, path):
    fig = plt.figure(figsize=(16,9))
    plt.plot(data)
    plt.xlabel("epochs")
    plt.ylabel(prop_name)
    plt.title("{}_vs_epochs".format(prop_name))
    plt.savefig(os.path.join(path,prop_name+".png"))


def make_dirs(path_list):
    for path in path_list:
        if not os.path.exists(path):
            os.mkdir(path)


def recall_prec(preds, gt):
    preds_ = (preds>0.5).float()
    # import ipdb; ipdb.set_trace()
    diff = preds_ - gt
    
    total_pos = torch.sum(preds_, (1,2,3))
    false_pos = torch.sum((diff==1), (1,2,3))
    true_pos = total_pos - false_pos

    total_negs = torch.sum((1 - preds_), (1,2,3))
    false_negs = torch.sum((diff==-1), (1,2,3))
    true_negs = total_negs - false_negs

    recall = (true_pos) / (true_pos + false_negs)
    precision = (true_pos) / (true_pos + false_pos)
   
    return recall, precision


def main(args):
    args = parse_args()
    ## data loader
    results_dir = os.path.join(os.getcwd(), "results")
    curr_run_dir = os.path.join(results_dir, datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p"))
    make_dirs([results_dir, curr_run_dir])    

    train_dataset = DirectionalFrontiers(root_dir=args.data_dir, 
        start_episode_num=3, max_episodes=1, max_count=500, transform=transforms.Compose([
            transforms.ToTensor()]))
    
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    num_epochs = args.num_epochs
    

    model = Network()
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    loss_fn = nn.MSELoss()

    loss_arr = []
    accuracy_arr = []
    recall_arr = []
    precision_arr = []
    for epoch in range(num_epochs):
        epoch_loss = 0.
        num_batches = 0
        epoch_recall = 0
        epoch_precision = 0
        for batch_id, ((trajectory_map, agent_dir, instruction), subset_frontier_matrix) in enumerate(dataloader):
        # for i in range(0,Y_train.shape[0], batch_size):
            num_batches += 1
            # X = X_train[i:min(Y_train.shape[0],i+batch_size)]
            # Y = Y_train[i:min(Y_train.shape[0],i+batch_size)]
            trajectory_map = trajectory_map.to(device)
            agent_dir = agent_dir.to(device)
            instruction = instruction.to(device)
            target_map = subset_frontier_matrix.to(device)

            output_map = model(trajectory_map, agent_dir, instruction)
            # import ipdb; ipdb.set_trace()

            loss = loss_fn(output_map, target_map)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            epoch_loss += loss.item()
            with torch.no_grad():
                recall, precision = recall_prec(output_map, target_map)

                epoch_recall += recall.mean().item()
                epoch_precision += precision.mean().item()
            
            if batch_id % args.log_interval == 0:
                print('====> Epoch: {}, batch: {}, Train Average loss: {:.4f}, recall: {:.4f}, precision: {:.4f}'.format(
                    epoch, batch_id, loss, recall.mean().item(), precision.mean().item()))
        
        loss_arr.append(epoch_loss/num_batches)
        recall_arr.append(epoch_recall / num_batches)
        precision_arr.append(epoch_precision/num_batches)
    plot_props(loss_arr, "loss", curr_run_dir)
    plot_props(recall_arr, "recall", curr_run_dir)
    plot_props(precision_arr, "precision", curr_run_dir)


if __name__ == "__main__":
    main(sys.argv)

    
