"""
Graph neural network models to learn fish dynamics
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

import json
import pickle as pkl
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


class EGNN(MessagePassing):
    """
    E(2) equivariant graph neural network to predict next time step position
    using only the current position.
    """
    def __init__(
            self,
            num_fish,
            batch_size,
            noise_std,
            mlp_hidden_dim,
            mlp_depth
    ):
        super().__init__(aggr='sum')
        self.noise_std = noise_std
        mlp_out = [mlp_hidden_dim for i in range(mlp_depth)] + [1]
        self.mlp = MLP(
            1,  # input squared distance
            mlp_out
        )

        # learning node representations TODO
        # embed_node = torch.zeros((num_fish, 1))
        # embed_node = torch.nn.init.xavier_uniform_(embed_node)
        # self.embed_node = torch.nn.parameter.Parameter(
        #     embed_node.repeat(batch_size, 1),
        #     requires_grad=True
        # )

    def forward(self, x, edge_index, edge_attr, pos):
        # noise graphs
        if self.training:
            edge_attr += torch.tensor(
                stats.norm(0, self.noise_std).rvs(edge_attr.shape),
                dtype=torch.float
            )

        # embed nodes TODO
        # x = x * self.embed_node

        # return an updated position
        return pos + self.propagate(
            edge_index=edge_index, x=x, edge_attr=edge_attr
        )

    def message(self, x_i, x_j, edge_attr):
        """
        Eq 3 of satorras et. al.
        """
        # put edges through an mlp
        return self.mlp(
            torch.sum(torch.square(edge_attr), axis=1).unsqueeze(1)
        )

    def aggregate(self, out, index, edge_attr, dim_size):
        """
        Eq 4 of satorras et. al.
        """
        m = edge_attr * out
        m = 1 / dim_size * self.aggr_module(m, index)
        return m


class EGNNVel(EGNN):
    """
    E(2) equivariant gnn with velocity incorporated
    """
    def __init__(
            self,
            noise_std,
            mlp_hidden_dim,
            mlp_depth
    ):
        super().__init__(aggr='sum')
        self.noise_std = noise_std
        mlp_out = [mlp_hidden_dim for i in range(mlp_depth)] + [1]
        self.mlp = MLP(
            2,  # input squared distance
            mlp_out
        )

    def forward(self, x, edge_index, edge_attr, pos, vel):
        # noise graphs
        if self.training:
            edge_attr += torch.tensor(
                stats.norm(0, self.noise_std).rvs(edge_attr.shape),
                dtype=torch.float
            )

        # return an updated position
        return pos + vel + self.propagate(  # TODO: make nodes learnable!
            edge_index=edge_index, x=x, edge_attr=edge_attr, vel=vel
        )

    def message(self, x_i, x_j, edge_attr):
        """
        Eq 3 of satorras et. al.
        """
        # put edges through an mlp
        return self.mlp(torch.square(edge_attr))  # NOTE: maybe sqrt this

    def aggregate(self, out, index, edge_attr, dim_size, vel):
        """
        Eq 4 of satorras et. al.
        """
        m = edge_attr * out
        m = 1 / dim_size * self.aggr_module(m, index)
        return m


if __name__ == '__main__':
    g = pkl.load(open('data/processed/8fish/240816f1.pkl', 'rb'))[0]
    config = json.load(open('model/config.json', 'r'))
    model = EGNNVel(
        noise_std=config['noise_std'],
        mlp_hidden_dim=config['mp_mlp_hidden_dim'],
        mlp_depth=config['mlp_depth']
    )

    out = model(
        x=g.x,
        edge_index=g.edge_index,
        edge_attr=g.edge_attr,
        pos=g.pos
    )
    print('done')
