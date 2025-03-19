"""
Script that evaluates a trained model
"""

import os
import glob
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

import torch
from torch_geometric.loader import DataLoader

import graph_utils.utils as gu
import graph_utils.nri_utils as gun

from model.gnn import MPNN, EGNN
from model.eggn_model import EGNN_vel
from model.NRI import MLPEncoder, MLPDecoder


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


def rollout_inference_nri(checkpoint, test_set):
    """
    runs inference with the nri model
    """
    # init model
    config = checkpoint['config']
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
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    encoder.eval()
    decoder.eval()
    
    num_fish = 8
    off_diag = np.ones([num_fish, num_fish]) - np.eye(num_fish)
    rel_rec = np.array(gun.encode_onehot(np.where(off_diag)[0]), dtype=float)
    rel_send = np.array(gun.encode_onehot(np.where(off_diag)[1]), dtype=float)
    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)

    # get data chunks
    test = gu.chunk_data_for_NRI(test_set, size=config['timesteps'])

    # loop through test data
    # This re-evaluates the graphs every chunk, but continues to predict
    # using the last predicted position
    preds = []
    last_pred = test[0].x.transpose(1, 2).contiguous()
    last_pred = last_pred[:,0::config['timesteps'],:,:].contiguous()
    for i in test:
        # evaluate graph
        logits = encoder(torch.FloatTensor(i.x).contiguous(), rel_rec, rel_send)
        edges = gun.gumbel_softmax(logits, tau=config['temp'], hard=config['hard'])
        prob = gun.my_softmax(logits, -1)

        # predict location
        curr_rel_type = edges.unsqueeze(1)
        for step in range(i.x.shape[2]):
            last_pred = decoder.single_step_forward(
                last_pred, rel_rec, rel_send, curr_rel_type
            )
            preds.append(last_pred)

    return torch.stack([p[:, :, :, :2].squeeze() for p in preds])


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
            ax.set_ylim([-1, 1])
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
    checkpoint = torch.load(model, map_location=torch.device('cpu'))
    datasets = checkpoint['datasets']
    config = checkpoint['config']

    # get data split
    data_list = []
    for dataset in datasets:
        data_list += pkl.load(open(dataset, 'rb'))

    train, val, test = gu.train_val_test(data_list)

    # getting val and test losses
    # loss_fn = torch.nn.MSELoss()
    # val_dataloader = DataLoader(val, batch_size=len(val))
    # test_dataloader = DataLoader(test, batch_size=len(test))
    # val_batch = next(iter(val_dataloader))
    # test_batch = next(iter(test_dataloader))

    # defining model and output
    # if checkpoint['model_type'] == 'mpnn':
    #     model = MPNN(
    #         noise_std=config['noise_std'],
    #         mp_mlp_hidden_dim=config['mp_mlp_hidden_dim'],
    #         update_mlp_hidden_dim=config['update_mlp_hidden_dim']
    #     )
    #     val_out = val_batch.acc
    #     test_out = test_batch.acc

    # elif checkpoint['model_type'] == 'egnn':
    #     model = EGNN(
    #         num_fish=train[0].x.shape[0],
    #         batch_size=len(test),  # NOTE
    #         noise_std=config['noise_std'],
    #         mlp_hidden_dim=config['mp_mlp_hidden_dim'],
    #         mlp_depth=config['mlp_depth']
    #     )
    #     val_out = val_batch.pos
    #     test_out = test_batch.pos

    # elif checkpoint['model_type'] == 'egnn_paper':
    #     model = EGNN_vel(
    #         in_node_nf=1,
    #         in_edge_nf=1,
    #         hidden_nf=64,
    #         device='cpu',
    #         n_layers=4,
    #         recurrent=True,
    #         norm_diff=False,
    #         tanh=False
    #     )
    #     val_out = val_batch.pos
    #     test_out = test_batch.pos

    # model.load_state_dict(checkpoint['model'])
    # model.eval()

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

    if 'encoder' and 'decoder' in checkpoint.keys():
        rollout_preds = rollout_inference_nri(checkpoint, test)
    elif checkpoint['model_type'] == 'mpnn':
        rollout_preds = rollout_inference_acc(model, test)
    elif checkpoint['model_type'] == 'egnn' or 'egnn_paper':
        rollout_preds = rollout_inference_pos(model, test, checkpoint['model_type'])

    # printing results
    # print('val loss:', val_loss)
    # print('test loss:', test_loss)
    plot_rollout(rollout_preds, test)


if __name__ == '__main__':
    saved_models = glob.glob('results/saved_models/*')
    most_recent = max(saved_models, key=os.path.getctime)
    evaluate(most_recent)
