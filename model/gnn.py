"""
Graph neural network model to learn fish dynamics
"""
# Notes:

# from kims talk: add noise into the inputs of the model (maybe will work).
# Intuition is that the model will learn something that is denoising as well
# as how things interact -> more stable rollouts over time.

# make an 'embedding' layer that converts nodes to a learned value. This
# can be thought of as the same way that the gnn is learning mass in the
# planet paper (? - ask brian about this)

# Only a one layer GNN

# In torch geometric temporal they have time lagged y values as node features.
# It could be good to make time lagged accelerations be the node features

from scipy import stats

import torch
from torchvision.ops import MLP
from torch_geometric.nn import MessagePassing


class MPNN(MessagePassing):
    """
    Generic expressive GNN with MLP update and message passing functions
    """
    def __init__(
            self,
            noise_std,
            mp_mlp_hidden_dim,
            update_mlp_hidden_dim,
    ):
        super().__init__(aggr='add')
        self.noise_std = noise_std

        # message passing mlp
        self.mp_mlp = MLP(
            2,  # accounting for edges
            [mp_mlp_hidden_dim, mp_mlp_hidden_dim, 1]
        )

        # update mlp
        self.update_mlp = MLP(
            1,
            [update_mlp_hidden_dim, update_mlp_hidden_dim, 2]
        )

    def forward(self, x, edge_index, edge_attr):
        # noise graphs
        if self.training:
            edge_attr += torch.tensor(
                stats.norm(0, self.noise_std).rvs(edge_attr.shape),
                dtype=torch.float
            )

        # get messages
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # put edges through an mlp
        return self.mp_mlp(edge_attr)

    def update(self, out):
        # update the nodes with aggregated edges
        return self.update_mlp(out)
