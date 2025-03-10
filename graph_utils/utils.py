"""
utilities to assist the training scrpit
"""

import numpy as np
import pickle as pkl
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data


def noise_graphs(data, std=0.2):
    """
    Adds noise to the edge attributes of each graph
    """
    noisy_graphs = []
    for g in tqdm(data, desc='noising graphs'):
        g.edge_attr += torch.tensor(
            stats.norm(0, std).rvs(g.edge_attr.shape),
            dtype=torch.float
        )
        noisy_graphs.append(g)

    return noisy_graphs


def vizualize_pyg_graph(graph):
    """
    plots a given pytorch geometric graph
    """
    pass


def plot_loss(loss_vals):
    fig, axs = plt.subplots()
    axs.plot(range(len(loss_vals)), loss_vals)
    axs.set_ylabel('loss')
    axs.set_xlabel('iter')


def train_val_test(data, train_size=0.8, val_size=0.1, test_size=0.1):
    """
    train on the first part and test on the second
    """
    train_idx = round(len(data) * train_size)
    val_idx = train_idx + round(len(data) * val_size)
    return data[:train_idx], data[train_idx:val_idx], data[val_idx:]


def vel_vel_corr(data):
    """
    calculates velocity-velocity auto-correlation values for each fish.
    Equationo taken from Vicsek and Zafeiris (2012)

    TODO: test this
    """
    c_vv = []

    # loop through time
    for i in range(len(data)):
        v_t = data[i].vel  # time

        # loop through previous values
        num = np.zeros_like(np.matmul(v_t, v_t.T).diag())
        denom = np.zeros_like(np.matmul(v_t, v_t.T).diag())
        for j in range(len(i + 1)):
            v_0 = data[j].vel.T
            num += np.matmul(v_t, v_0.T).diag()
            denom += np.matmul(v_0, v_0.T).diag()

        # avg over starting times
        num /= num.shape[0]
        denom /= denom.shape[0]

        # calc correlation at time t
        c = np.sum(num / denom) / v_t.shape[0]

        # store
        c_vv.append(c)

    return c_vv


def chunk_data_for_NRI(data, size=49):
    """
    Splits data for the NRI model
    """
    # Making data divisible by size
    data_rm = len(data) % size
    rm_left = data_rm // 2
    rm_right = data_rm - rm_left
    data = data[rm_left:-rm_right]

    # loop through data
    chunked_graphs = []
    for i in range(len(data) // size):
        start = i * size
        stop = start + size
        node_emb = [
            torch.cat((j.pos, j.vel), axis=1) for j in data[start:stop]
        ]
        graph = Data(
            x=torch.einsum('ijkl->ikjl', torch.stack(node_emb).unsqueeze(0)),
            edge_index=data[i].edge_index,
        )
        chunked_graphs.append(graph)

    return chunked_graphs


if __name__ == '__main__':
    fp = 'data/fish/processed/8fish/240816f1.pkl'
    data = pkl.load(open(fp, 'rb'))
    vel_vel_corr(data)