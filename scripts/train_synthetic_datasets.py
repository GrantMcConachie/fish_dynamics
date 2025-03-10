"""
Training script for models on synthetic data
"""

import numpy as np

from graph_utils.preprocess import generate_graphs
from synthetic_data_generation.datasets import SpringSim


def generate_data(train_p=0.8):
    """
    Generate a train and test set for a large 
    """
    # dataset
    sim = SpringSim()
    # time = 10000000
    time = 5000
    sample_freq = 100

    # sampling
    loc, vel, edges = sim.sample_trajectory(
        T=time,
        sample_freq=sample_freq
    )

    # making pytorch geometric data
    state_vars = {}
    state_vars['x'] = np.einsum('ijk->kji', loc)
    state_vars['x_dot'] = np.einsum('ijk->kji', vel)
    state_vars['x_dot_dot'] = np.zeros_like(np.einsum('ijk->kji', vel))

    # generate graphs
    graphs = generate_graphs(state_vars)

    # splitting to train and test
    train_cut = round(loc.shape[0] * train_p)
    train_graphs = graphs[:train_cut]
    test_graphs = graphs[train_cut:]

    return train_graphs, test_graphs, gt_edges


def train(data):
    pass


if __name__ == '__main__':
    train_loc, train_vel, gt_edges = generate_data()
    train(train_loc, train_vel)
