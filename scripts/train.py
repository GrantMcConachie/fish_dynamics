"""
Training script for the GNN
"""

import os
import json
import time
import numpy as np
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch_geometric.loader import DataLoader

from model.gnn import MPNN, EGNN
import graph_utils.utils as gu


def train(datasets, plot_loss=True, save_model=True, model_type='egnn'):
    # load datasets of interest
    data_list = []
    for dataset in datasets:
        data_list += pkl.load(open(dataset, 'rb'))

    # split into train and test
    train, val, test = gu.train_val_test(data_list)

    # get model config
    config = json.load(open('model/config.json', 'r'))

    # init model
    if model_type == 'mpnn':
        model = MPNN(
            noise_std=config['noise_std'],
            mp_mlp_hidden_dim=config['mp_mlp_hidden_dim'],
            update_mlp_hidden_dim=config['update_mlp_hidden_dim']
        )

    elif model_type == 'egnn':
        model = EGNN(
            num_fish=train[0].x.shape[0],
            batch_size=config['batch_size'],
            noise_std=config['noise_std'],
            mlp_hidden_dim=config['mp_mlp_hidden_dim'],
            mlp_depth=config['mlp_depth']
        )

    model.train()
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=config['adam_lr'])

    # training loop
    loss_vals = []
    dataloader = DataLoader(
        train,
        batch_size=config['batch_size'],
        shuffle=True
    )
    for i in tqdm(range(config['epochs'])):
        epoch_loss = []
        for g in dataloader:
            opt.zero_grad()

            # foraward pass
            out = model(
                x=g.x,
                edge_index=g.edge_index,
                edge_attr=g.edge_attr,
                pos=g.pos
            )

            if model_type == 'mpnn':
                loss = loss_fn(out, g.acc)

            elif model_type == 'egnn':
                loss = loss_fn(out, g.pos_next)

            # backward pass
            loss.backward()
            opt.step()

            # record keeping
            loss_vals.append(loss.item())
            epoch_loss.append(loss.item())

        # loss per epoch
        print(f'epoch {i} loss: {np.mean(epoch_loss)}')

    # post processing model
    if plot_loss:
        gu.plot_loss(loss_vals)
        gu.plot_loss(epoch_loss)
        plt.show()

    if save_model:
        fp = 'results/saved_models'  # TODO: add option to specify model name
        file = os.path.join(fp, f'tmp{round(time.time())}.pt')
        torch.save(
            {
                'model': model.state_dict(),
                'datasets': datasets,
                'config': config,
                'loss': loss_vals,
                'model_type': model_type
            },
            file
        )


if __name__ == '__main__':
    datasets = ['data/processed/8fish/240816f1.pkl']
    train(datasets=datasets)
