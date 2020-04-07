import numpy as np
import os
import torch 
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys, os
from human_to_robot_network import *
from dataset import *
from datatime import datetime
from code.dataset import DirectionalFrontiers

import torch.utils.data.DataLoader as DataLoader
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, 
        help='learning rate')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, 
        help='batch_size')
    parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=60, 
        help='number of epochs')
    parser.add_argument('--data-dir', dest='data_dir', type=str, default='train_data', 
        help='path to data files')

    return parser.parse_args()


def plot_props(data, prop_namen, path):
    fig = plt.figure(figsize=(16,9))
    plt.plot(data)
    plt.xlabel("epochs")
    plt.ylabel(prop_name)
    plt.title("{}_vs_epochs".format(prop_name))
    plt.savefig(os.path.join(path,prop_name+".png"))


def make_dirs(path_list):
    for path in path_list:
        if not os.path.exists():
            os.mkdir(path)


def recall_prec(preds, gt):
    preds_ = (preds>0.5)
    diff = preds - gt
    
    total_pos = torch.sum(preds, (1,2))
    false_pos = torch.sum((diff==1), (1,2))
    true_pos = total_pos - false_pos

    total_negs = torch.sum((1 - preds), (1,2))
    false_negs = torch.sum((diff==-1), (1,2))
    true_negs = total_negs - false_negs

    recall = (true_pos) / (true_pos + false_negs)
    precision = (true_pos) / (true_pos + false_pos)
   
    return recall, precision


def main(args):
    args = parse_args()
    ## data loader
    
    results_dir = os.path.join(os.cwd(), datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p"))
    make_dirs([results_dir])    

    dataset = DirectionalFrontiers(root_dir=args.data_dir, 
        start_episode_num=3, max_episodes=3, max_count=500, transform=transforms.ToTensor)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # data_iter = iter(dataset)
    num_epochs = args.num_epochs
    
    # batch_size = args.batch_size 
    
    # X_train = []
    # Y_train = []

    model = network(args.lr)
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    loss_fn = nn.MSELoss()

    loss_arr = []
    accuracy_arr = []
    recall_arr = []
    precision_arr = []
    for epoch in num_epochs:
        epoch_loss = 0.
        num_batches = 0
        epoch_recall = 0
        epoch_precision = 0
        for batch_id, data in enumerate(dataloader):
        # for i in range(0,Y_train.shape[0], batch_size):
            # num_batches += 1
            # X = X_train[i:min(Y_train.shape[0],i+batch_size)]
            # Y = Y_train[i:min(Y_train.shape[0],i+batch_size)]
            x_map = data[0].to(device)
            agent_dir = data[1].to(device)
            instruction = data[2].to(device)
            target_map = data[3].to(device)

            output = model(x_map, agent_dir, instruction)
            
            loss = loss_fn(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            with torch.no_grad():
                recall, precision = recall_prec(pred, Y_train)

                epoch_recall += recall.mean().item()
                epoch_precision += precision.mean().item()

        
        loss_arr.append(epoch_loss/num_batches)
        recall_arr.append(epoch_recall / num_batches)
        precision_arr.append(epoch_precision/num_batches)
    plot_props(loss_arr, "loss", results_dir)
    plot_props(recall_arr, "recall", results_dir)
    plot_props(precision_arr, "precision", results_dir)

if __name__ == "__main__":
    main(sys.argv)

    
