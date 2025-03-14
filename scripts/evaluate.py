"""
Script that evaluates a trained model
"""

import os
import glob
import pickle as pkl
import matplotlib.pyplot as plt

import torch
from torch_geometric.loader import DataLoader

from model.gnn import MPNN, EGNN
from model.eggn_model import EGNN_vel
import graph_utils.utils as gu


def rollout_inference_acc(model, test_set, frame_rate=120):
    """
    Predicts the position of the fish over time with a model that predicts
    acceleration values
    """
    # initial values
    pos = test_set[0].pos
    vel = test_set[0].vel
    edge_index = test_set[0].edge_index

    # initial prediction
    pred_acc = model(
        x=test_set[0].x,
        edge_index=test_set[0].edge_index,
        edge_attr=test_set[0].edge_attr,
        pos=test_set[0].pos
    )

    # rollout prediction
    rollout_preds_pos = []
    rollout_preds_vel = []
    rollout_preds_acc = []
    for i in range(len(test_set)):
        rollout_preds_acc.append(pred_acc)
        vel = vel + pred_acc / frame_rate
        pos = pos + vel / frame_rate
        rollout_preds_pos.append(pos)
        rollout_preds_vel.append(vel)

        # predict next step
        new_edge_attr = torch.tensor(
                pos[edge_index[0]] - pos[edge_index[1]],
                dtype=torch.float
            )
        pred_acc = model(
            x=test_set[0].x,
            edge_index=test_set[0].edge_index,
            edge_attr=new_edge_attr
        )

    return [
        rollout_preds_pos,
        rollout_preds_vel,
        rollout_preds_acc
    ]


def rollout_inference_pos(model, test_set, model_type, frame_rate=120):
    """
    Predicts the position of the fish over time with a model that predicts the
    position of the fish
    """
    # initial values
    pos = test_set[0].pos.detach().clone()
    vel = test_set[0].vel.detach().clone()
    edge_index = test_set[0].edge_index.detach().clone()
    edge_attr = test_set[0].edge_attr.detach().clone()

    # initial prediction
    if model_type == 'egnn_paper':
        nodes = torch.sqrt(
            torch.sum(vel ** 2, dim=1)
        ).unsqueeze(1).detach()
        pos_next = model(
            nodes,
            pos.detach().clone(),
            [edge_index[0], edge_index[1]],
            vel,
            torch.sum((edge_attr) ** 2, axis=1).unsqueeze(1)
        )
        vel = (pos_next - pos) / frame_rate
        pos = pos_next

    else:
        pos = model(
            x=test_set[0].x,
            edge_index=test_set[0].edge_index,
            edge_attr=test_set[0].edge_attr,
            pos=test_set[0].pos
        )

    # rollout prediction
    rollout_preds_pos = []
    rollout_preds_vel = []
    rollout_preds_acc = []
    for i in range(len(test_set)):
        rollout_preds_pos.append(pos)

        # predict next step
        new_edge_attr = torch.tensor(
                pos[edge_index[0]] - pos[edge_index[1]],
                dtype=torch.float
            )
        
        if model_type == 'egnn_paper':
            nodes = torch.sqrt(
                torch.sum(vel ** 2, dim=1)
            ).unsqueeze(1).detach()
            pos_next = model(
                nodes,
                pos.detach().clone(),
                [edge_index[0], edge_index[1]],
                vel,
                torch.sum((new_edge_attr) ** 2, axis=1).unsqueeze(1)
            )
            vel = (pos_next - pos) / frame_rate
            pos = pos_next

        else:
            pos = model(
                x=test_set[0].x,  # all ones for now
                edge_index=test_set[0].edge_index,
                edge_attr=new_edge_attr,
                pos=pos
            )

    return [
        rollout_preds_pos,
        rollout_preds_vel,
        rollout_preds_acc
    ]


def plot_rollout(rollout_preds, test_set):
    num_fish = rollout_preds[0].shape[0]
    for i in range(num_fish // 2):
        fig, axs = plt.subplots(4, 1)

        fish_1x_p = [r[i * 2][0].detach().numpy() for r in rollout_preds]
        fish_1y_p = [r[i * 2][1].detach().numpy() for r in rollout_preds]
        fish_2x_p = [r[i * 2 + 1][0].detach().numpy() for r in rollout_preds]
        fish_2y_p = [r[i * 2 + 1][1].detach().numpy() for r in rollout_preds]

        fish_1x_t = [r.pos[i * 2][0].detach().numpy() for r in test_set]
        fish_1y_t = [r.pos[i * 2][1].detach().numpy() for r in test_set]
        fish_2x_t = [r.pos[i * 2 + 1][0].detach().numpy() for r in test_set]
        fish_2y_t = [r.pos[i * 2 + 1][1].detach().numpy() for r in test_set]

        axs[0].plot(range(len(fish_1x_p)), fish_1x_p, label='preds', alpha=0.4)
        axs[0].plot(range(len(fish_1x_t)), fish_1x_t, label='truth', alpha=0.4)
        axs[0].set_title(f'fish {i * 2} x')
        axs[1].plot(range(len(fish_1y_p)), fish_1y_p, alpha=0.4)
        axs[1].plot(range(len(fish_1y_t)), fish_1y_t, alpha=0.4)
        axs[1].set_title(f'fish {i * 2} y')
        axs[2].plot(range(len(fish_2x_p)), fish_2x_p, alpha=0.4)
        axs[2].plot(range(len(fish_2x_t)), fish_2x_t, alpha=0.4)
        axs[2].set_title(f'fish {i * 2 + 1} x')
        axs[3].plot(range(len(fish_2y_p)), fish_2y_p, alpha=0.4)
        axs[3].plot(range(len(fish_2y_t)), fish_2y_t, alpha=0.4)
        axs[3].set_title(f'fish {i * 2 + 1} y')

        for ax in axs.reshape(-1):
            ax.set_ylim([-150, 150])
            # ax.set_xlim([0, 240])
            pass

        fig.supxlabel('time')
        fig.supylabel('position')
        fig.legend()
        plt.tight_layout()

    plt.show()


def evaluate(model):
    """
    Returns validation loss, test loss, and a rollout inference of the test set
    """
    # load model
    checkpoint = torch.load(model)
    datasets = checkpoint['datasets']
    config = checkpoint['config']

    # get data split
    data_list = []
    for dataset in datasets:
        data_list += pkl.load(open(dataset, 'rb'))

    train, val, test = gu.train_val_test(data_list)

    # getting val and test losses
    loss_fn = torch.nn.MSELoss()
    val_dataloader = DataLoader(val, batch_size=len(val))
    test_dataloader = DataLoader(test, batch_size=len(test))
    val_batch = next(iter(val_dataloader))
    test_batch = next(iter(test_dataloader))

    # defining model and output
    if checkpoint['model_type'] == 'mpnn':
        model = MPNN(
            noise_std=config['noise_std'],
            mp_mlp_hidden_dim=config['mp_mlp_hidden_dim'],
            update_mlp_hidden_dim=config['update_mlp_hidden_dim']
        )
        val_out = val_batch.acc
        test_out = test_batch.acc

    elif checkpoint['model_type'] == 'egnn':
        model = EGNN(
            num_fish=train[0].x.shape[0],
            batch_size=len(test),  # NOTE
            noise_std=config['noise_std'],
            mlp_hidden_dim=config['mp_mlp_hidden_dim'],
            mlp_depth=config['mlp_depth']
        )
        val_out = val_batch.pos
        test_out = test_batch.pos

    elif checkpoint['model_type'] == 'egnn_paper':
        model = EGNN_vel(
            in_node_nf=1,
            in_edge_nf=1,
            hidden_nf=64,
            device='cpu',
            n_layers=4,
            recurrent=True,
            norm_diff=False,
            tanh=False
        )
        val_out = val_batch.pos
        test_out = test_batch.pos

    model.load_state_dict(checkpoint['model'])
    model.eval()

    # val_loss = loss_fn(
    #     model(
    #         x=val_batch.x,
    #         edge_index=val_batch.edge_index,
    #         edge_attr=val_batch.edge_attr,
    #         pos=val_batch.pos
    #     ),
    #     val_out
    # )
    # test_loss = loss_fn(
    #     model(
    #         x=test_batch.x,
    #         edge_index=test_batch.edge_index,
    #         edge_attr=test_batch.edge_attr,
    #         pos=test_batch.pos
    #     ),
    #     test_out
    # )

    if checkpoint['model_type'] == 'mpnn':
        rollout_preds = rollout_inference_acc(model, test)
    elif checkpoint['model_type'] == 'egnn' or 'egnn_paper':
        rollout_preds = rollout_inference_pos(model, test, checkpoint['model_type'])

    # printing results
    # print('val loss:', val_loss)
    # print('test loss:', test_loss)
    plot_rollout(rollout_preds[0], test)


if __name__ == '__main__':
    saved_models = glob.glob('results/saved_models/*')
    most_recent = max(saved_models, key=os.path.getctime)
    evaluate(most_recent)
