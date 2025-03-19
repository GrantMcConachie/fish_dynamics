"""
A script for training the NRI model (Kipf et. al. 2018) on fish data
"""

import os
import json
import time
import numpy as np
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch_geometric.loader import DataLoader

from model.gnn import MPNN, EGNN
from model.eggn_model import EGNN_vel
from model.NRI import MLPEncoder, MLPDecoder

import graph_utils.utils as gu
import graph_utils.nri_utils as gun


def validate(encoder, decoder, rel_rec, rel_send, val, config, num_fish, device):
    """
    gets validation loss of the model
    """
    val_loss = []
    val_loss_nll = []
    val_loss_kl = []
    val_dataloader = DataLoader(
        val,
        batch_size=config['batch_size'],
        shuffle=False
    )

    # val loop
    for g in val_dataloader:
        g.to(device)
        # forward pass
        logits = encoder(g.x, rel_rec, rel_send)
        edges = gun.gumbel_softmax(logits, tau=config['temp'], hard=config['hard'])
        prob = gun.my_softmax(logits, -1)
        output = decoder(
                g.x, edges, rel_rec, rel_send, config['prediction_steps']
            )

        # calculate loss
        target = g.x[:, :, 1:, :]  # No prediction for 0
        loss_nll = gun.nll_gaussian(output, target, config['var'])
        loss_kl = gun.kl_categorical_uniform(
            prob, num_fish, config['edge_types']
        )
        loss = loss_nll + loss_kl
        val_loss.append(loss.item())
        val_loss_nll.append(loss_nll.item())
        val_loss_kl.append(loss_kl.item())

    return np.mean(val_loss), np.mean(val_loss_nll), np.mean(val_loss_kl)


def train(datasets, plot_loss=True, save_model=True, model_type='egnn'):
    """
    training loop for the model
    """
    # check for gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get model config
    config = json.load(open('model/config_NRI.json', 'r'))

    # load datasets of interest
    train_list = []
    val_list = []
    test_list = []
    for dataset in datasets:
        data_list = pkl.load(open(dataset, 'rb'))

        # split into train and test
        train, val, test = gu.train_val_test(data_list)

        # split train val test into chunks TODO: incorporate different chunk every epoch
        train_list += gu.chunk_data_for_NRI(train, size=config['timesteps'])
        val_list += gu.chunk_data_for_NRI(val, size=config['timesteps'])

    # init model
    encoder = MLPEncoder(
        config['timesteps'] * config['node_dim'],
        config['encoder_hidden'],
        config['edge_types'],
        config['encoder_dropout'],
        config['factor']
    )
    decoder = MLPDecoder(
        n_in_node=config['node_dim'],
        edge_types=config['edge_types'],
        msg_hid=config['decoder_hidden'],
        msg_out=config['decoder_hidden'],
        n_hid=config['decoder_hidden'],
        do_prob=config['decoder_dropout'],
        skip_first=config['skip_first']
    )
    encoder.to(device)
    decoder.to(device)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=config['lr']
    )
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=config['lr_decay'], gamma=config['gamma']
    )
    scheduler.step()

    # set up fully connected graph
    num_fish = train_list[0].x.shape[1]
    off_diag = np.ones([num_fish, num_fish]) - np.eye(num_fish)
    rel_rec = np.array(gun.encode_onehot(np.where(off_diag)[0]), dtype=float)
    rel_send = np.array(gun.encode_onehot(np.where(off_diag)[1]), dtype=float)
    rel_rec = torch.FloatTensor(rel_rec).to(device)
    rel_send = torch.FloatTensor(rel_send).to(device)

    # training loop
    tot_train_loss = []
    tot_val_loss = []
    best_val_loss = 1e10
    dataloader = DataLoader(
        train_list,
        batch_size=config['batch_size'],
        shuffle=True
    )
    for i in tqdm(range(config['epochs'])):
        encoder.train()
        decoder.train()
        epoch_loss = []
        epoch_loss_nll = []
        epoch_loss_kl = []

        for g in dataloader:
            g.to(device)
            optimizer.zero_grad()

            # encoder
            logits = encoder(g.x, rel_rec, rel_send)

            # sample
            edges = gun.gumbel_softmax(logits, tau=config['temp'], hard=config['hard'])
            prob = gun.my_softmax(logits, -1)

            # decoder
            output = decoder(
                g.x, edges, rel_rec, rel_send, config['prediction_steps']
            )

            # calculate loss
            target = g.x[:, :, 1:, :]  # No prediction for 0
            loss_nll = gun.nll_gaussian(output, target, config['var'])
            loss_kl = gun.kl_categorical_uniform(
                prob, num_fish, config['edge_types']
            )
            loss = loss_nll + loss_kl

            # backward pass
            loss.backward()
            optimizer.step()

            # record keeping
            epoch_loss.append(loss.item())
            epoch_loss_nll.append(loss_nll.item())
            epoch_loss_kl.append(loss_kl.item())

        # get validation loss
        encoder.eval()
        decoder.eval()
        curr_val_loss, val_loss_nll, val_loss_kl = validate(
            encoder, decoder, rel_rec, rel_send, val_list, config, num_fish, device
        )

        # epoch metrics
        print(
            f'epoch {i}\n',
            f'train loss: {np.mean(epoch_loss)}\n',
            f'train nll loss: {np.mean(epoch_loss_nll)}\n',
            f'train kl loss: {np.mean(epoch_loss_kl)}\n',
            f'val loss: {curr_val_loss}\n',
            f'val nll loss: {val_loss_nll}\n',
            f'val kl loss: {val_loss_kl}\n\n'
        )

        # save model
        if save_model and curr_val_loss < best_val_loss:
            tot_train_loss.append(np.mean(epoch_loss))
            tot_val_loss.append(curr_val_loss)
            fp = 'results/saved_models'
            file = os.path.join(fp, f'nri_all8fish_nojumps.pt')
            torch.save(
                {
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'datasets': datasets,
                    'config': config,
                    'train_loss': tot_train_loss,
                    'val_loss': tot_val_loss,
                    'model_type': model_type
                },
                file
            )
            best_val_loss = curr_val_loss


if __name__ == '__main__':
    datasets = [
        'data/fish/processed/8fish/240816f2.pkl',
        'data/fish/processed/8fish/240816f4.pkl',
        'data/fish/processed/8fish/240820f2.pkl',
        'data/fish/processed/8fish/240820f4.pkl',
        'data/fish/processed/8fish/240821f2.pkl',
        'data/fish/processed/8fish/240821f3.pkl',
        'data/fish/processed/8fish/240821f5.pkl',
        'data/fish/processed/8fish/240821f7.pkl'
    ]
    train(datasets=datasets)
