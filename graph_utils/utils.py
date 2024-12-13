"""
utilities to assist the training scrpit
"""

from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt

import torch


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
    plt.show()


def train_val_test(data, train_size=0.8, val_size=0.1, test_size=0.1):
    """
    train on the first part and test on the second
    """
    train_idx = round(len(data) * train_size)
    val_idx = train_idx + round(len(data) * val_size)
    return data[:train_idx], data[train_idx:val_idx], data[val_idx:]
