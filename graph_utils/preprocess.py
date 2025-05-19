"""
Script that preprocesses the raw data

TODO: encode heading angle
"""

import os
import h5py
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm

import torch

from torch_geometric.data import Data


def generate_x_dot(loc, frame_rate=120):
    """
    generates x dot data from location data
    """
    # init
    x = []
    x_dot = []
    x_dot_dot = []

    for i, fish in enumerate(loc):
        # averaging over keypoints
        pos = np.mean(fish, axis=1)

        # linearaly interpolating over nans
        pos = np.array(
            pd.DataFrame(pos).interpolate(axis=1, limit_direction='both')
        )

        # px to mm, center, and normalize
        pos *= (300 / 750)
        pos -= 150
        pos /= 150

        # calculating velocity and acceleration
        vel = np.array(
            (np.convolve([1, -1], pos[0, :], 'valid'),
             np.convolve([1, -1], pos[1, :], 'valid'))
        ) * frame_rate

        acc = np.array(
            (np.convolve([1, -1], vel[0, :], 'valid'),
             np.convolve([1, -1], vel[1, :], 'valid'))
        ) * frame_rate

        x.append(pos)
        x_dot.append(vel)
        x_dot_dot.append(acc)

    return {
        'x': np.array(x),
        'x_dot': np.array(x_dot),
        'x_dot_dot': np.array(x_dot_dot)
    }


def make_edge_idx(num_nodes):
    """
    makes a edge index for a fully connected graph given a number of nodes.
    Ignores self edges
    """
    arr = np.arange(num_nodes)
    edge_idx = np.array(np.meshgrid(arr, arr)).T.reshape(-1, 2).T
    self_edge = np.where(edge_idx[0] == edge_idx[1])

    return torch.tensor(
        np.delete(edge_idx, self_edge, axis=1),
        dtype=torch.long
    )


def generate_graphs(state_vars):
    """
    Takes position data and generates a pytorch geometric graph

    Input:
        state_vars (dict) - fish position, velocity, and acceleration data for
          a given video. (num_units, x-y position, time)

    Return:
        graphs (list) - list of pytorch gemoetric graphs
    """
    # learnable node matrix
    num_nodes = state_vars['x'].shape[0]
    node_feats = torch.ones(size=(num_nodes, 1))

    # edge index for fully connected graph
    edge_index = make_edge_idx(num_nodes)

    # adjust if all the same value
    if state_vars['x_dot_dot'].shape[-1] == state_vars['x'].shape[-1]:
        state_vars['x_dot_dot'] = state_vars['x_dot_dot'][:, :, :-1]

    # loop through time
    graphs = []
    for i in tqdm(range(state_vars['x'].shape[-1])):
        # Truncating x, so every position has an acceleration
        if i == state_vars['x_dot_dot'].shape[-1]:
            break

        # making edge features
        pos = state_vars['x'][:, :, i]
        pos_next = state_vars['x'][:, :, i+1]
        edge_attr = torch.tensor(  # relative positions
            pos[edge_index[0]] - pos[edge_index[1]],
            dtype=torch.float
        )

        # making graph
        graph = Data(
            x=node_feats,
            edge_attr=edge_attr,
            edge_index=edge_index,
            pos=torch.tensor(pos, dtype=torch.float),
            pos_next=torch.tensor(pos_next, dtype=torch.float),
            vel=torch.tensor(state_vars['x_dot'][:, :, i], dtype=torch.float),
            acc=torch.tensor(
                state_vars['x_dot_dot'][:, :, i], dtype=torch.float
            )
        )
        graphs.append(graph)

    return graphs


def save_graphs(f, fp, graphs):
    """
    Saving all generated graphs

    f (str) - data file
    fp (str) - parent directory for file
    graphs (list) - All generated graphs from given file
    """
    save_path = os.path.join(fp, f)
    pkl.dump(graphs, open(save_path, 'wb'))


def main(fp, save_fp):
    """
    Gets location data out of the raw files and saves them as pkl files
    """
    # list files
    fish_files = os.listdir(fp)

    # loop though files
    graph_lens = []
    for f in fish_files:
        fish_h5 = h5py.File(os.path.join(fp, f))

        # take out location data
        fish_hdf = fish_h5['tracks']

        # generate state vectors
        state_vectors = generate_x_dot(fish_hdf)

        # generate graphs
        graphs = generate_graphs(state_vectors)

        # saving
        f = f.replace('.h5', '.pkl')
        save_graphs(f, save_fp, graphs)


if __name__ == '__main__':
    # 8 fish
    main(
        fp='data/fish/raw data/8fish',
        save_fp='data/fish/processed/8fish'
    )

    # 10 fish
    main(
        fp='data/fish/raw data/10fish/DATA',
        save_fp='data/fish/processed/10fish'
    )
