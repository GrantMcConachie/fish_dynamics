"""
Script that evaluates a trained model
"""

import os
import glob
import numpy as np
import pickle as pkl
from matplotlib import cm
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data
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

    tot_mse = []
    tot_pred_edges = []
    for dataset in test_set:
        # get data chunks
        test = gu.chunk_data_for_NRI(dataset, size=config['timesteps'])

        # loop through test data
        # This re-evaluates the graphs every chunk, but continues to predict
        # using the last predicted position
        dat_mse = []
        pred_edges = []
        last_pred = test[0].x.transpose(1, 2).contiguous()
        last_pred = last_pred[:,0::config['timesteps'],:,:].contiguous()
        for itr, i in enumerate(test):
            # evaluate graph
            logits = encoder(torch.FloatTensor(i.x).contiguous(), rel_rec, rel_send)
            edges = gun.gumbel_softmax(logits, tau=config['temp'], hard=config['hard'])
            prob = gun.my_softmax(logits, -1)
            pred_edges.append(edges)

            # predict location
            curr_rel_type = prob.unsqueeze(1)
            for step in range(i.x.shape[2]):
                last_pred = decoder.single_step_forward(
                    last_pred, rel_rec, rel_send, curr_rel_type
                )
                # calculate mse
                if i.x.shape[2] - 1 == step:
                    try:
                        mse = torch.nn.functional.mse_loss(last_pred[:, :, :, :2].squeeze(), test[itr+1].x[:, :, 0, :2].squeeze()).detach().numpy()
                    except:
                        mse = np.array(0)
                else:
                    mse = torch.nn.functional.mse_loss(last_pred[:, :, :, :2].squeeze(), i.x[:, :, step+1, :2].squeeze()).detach().numpy()

                dat_mse.append(mse)
        
        # append for all datasets
        tot_mse.append(dat_mse)
        tot_pred_edges.append(pred_edges)

    return tot_mse, tot_pred_edges


def plot_mse_over_time(mse_over_time):
    smallest_dat = min([len(i) for i in mse_over_time])

    # make array out of mse error
    mse_dat = np.array([i[:smallest_dat] for i in mse_over_time])
    avg_vals = np.mean(mse_dat, axis=0)
    std_vals = np.std(mse_dat, axis=0)

    # plot a bar graph
    fig, axs = plt.subplots()
    axs.plot(range(len(avg_vals)), avg_vals, linestyle='--', color="#CC0000")
    axs.fill_between(range(len(avg_vals)), avg_vals-std_vals, avg_vals+std_vals, alpha=0.2, color="#CC0000")
    axs.set_xlabel("# of rollout frames")
    axs.set_ylabel("mse")
    axs.set_title("Test Set Rollout Inference")
    axs.set_ylim([0, 1])
    axs.set_xlim([0, 3600])
    plt.show()


def plot_rollout_like_NRI(checkpoint, test_set):
    """
    Plots similar to the spring mass system where the first 49 are the learned
    graph and the next 49 are the rollout predictions.
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

    # loop through datasets
    for dataset in test_set:
        test = gu.chunk_data_for_NRI(dataset, size=config['timesteps'])
        for itr, graphs in enumerate(test):
            if itr == len(dataset) - 1:
                pass

            else:
                # predict the edges
                logits = encoder(torch.FloatTensor(graphs.x).contiguous(), rel_rec, rel_send)
                edges = gun.gumbel_softmax(logits, tau=config['temp'], hard=config['hard'])
                prob = gun.my_softmax(logits, -1)

                # predict the next n timesteps given the edges
                output = decoder(torch.FloatTensor(test[itr+1].x).contiguous(), edges, rel_rec, rel_send, 49)

                # plotting
                num_fish = 8
                start = 0.0
                stop = 1.0
                num_colors = num_fish
                cm_subsection = np.linspace(start, stop, num_colors) 

                colors = [cm.Set1(x) for x in cm_subsection]
                fig, axs = plt.subplots(1, 2)

                for i in range(num_fish):
                    for t in range(graphs.x.shape[2]):
                        # Plot fading tail for past locations.
                        axs[1].plot(graphs.x[:, i, t, 0].detach().numpy(), graphs.x[:, i, t, 1].detach().numpy(), '.', markersize=5, 
                                color=colors[i], alpha=0.2)
                        axs[0].plot(graphs.x[:, i, t, 0].detach().numpy(), graphs.x[:, i, t, 1].detach().numpy(), '.', markersize=5, 
                                color=colors[i], alpha=0.2)
                        
                    for t in range(test[itr+1].x.shape[2]):
                        axs[1].plot(test[itr+1].x[:, i, t, 0].detach().numpy(), test[itr+1].x[:, i, t, 1].detach().numpy(), '.', markersize=5,
                                    color=colors[i], alpha=1.0)
                        
                    for t in range(output.shape[2]):
                        axs[0].plot(output[:, i, t, 0].detach().numpy(), output[:, i, t, 1].detach().numpy(), '.', markersize=5,
                                    color=colors[i], alpha=1.0)
                        
                    # Plot final location.
                    axs[1].plot(test[itr+1].x[:, i, -1, 0], test[itr+1].x[:, i, -1, 1], 'o', color=colors[i])
                    axs[0].plot(output[:, i, -1, 0].detach().numpy(), output[:, i, -1, 1].detach().numpy(), 'o', color=colors[i])

                axs[0].set_ylim([-1, 1])
                axs[1].set_ylim([-1, 1])
                axs[0].set_xlim([-1, 1])
                axs[1].set_xlim([-1, 1])
                axs[0].set_title('Predicted')
                axs[1].set_title('Ground Truth')

                plt.show()


def plot_predicted_edges_over_time(checkpoint):
    # load in dataset of interest
    data = pkl.load(open('data/fish/processed/8fish/240816f1.pkl', 'rb'))
    chuncked_data = gu.chunk_data_for_NRI(data)
    
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

    # loop through chunked data and predict edges
    pred_graphs = []
    len_of_edges = []
    sen_rec_info = []
    neighbors = []
    for graphs in chuncked_data:
        # edges correspond to rel_rec and rel_send [0, 1] == edge, [1, 0] == no edge
        logits = encoder(torch.FloatTensor(graphs.x).contiguous(), rel_rec, rel_send)
        edges = gun.gumbel_softmax(logits, tau=config['temp'], hard=config['hard'])
        prob = gun.my_softmax(logits, -1)

        # find avg position of each fish
        avg_pos = [torch.mean(graphs.x[:, i, :, :2], dim=1) for i in range(num_fish)]
        avg_vel = [torch.mean(graphs.x[:, i, :, 2:], dim=1) for i in range(num_fish)]

        # if there is an edge between nodes, find the distance between fish and
        # record
        for i, edge in enumerate(edges.squeeze()):
            is_edge = torch.argmax(edge) # 1 if there is an edge
            if is_edge:
                sender = graphs.edge_index[0, i].detach().numpy()
                reciever = graphs.edge_index[1, i].detach().numpy()

                # find distance where there are edges
                len_of_edges.append(
                    torch.linalg.norm(avg_pos[sender] - avg_pos[reciever]).detach().numpy()
                )
                sen_rec_info.append((sender, reciever))

                # finding neighbor info
                # find orientation
                theta = np.arctan(avg_vel[reciever][:, 1] / avg_vel[reciever][:, 0])
                if avg_vel[reciever][:, 0] < 0:
                    theta += np.pi
                rot_mat = np.array(
                    [[np.cos(np.pi / 2 - theta), -np.sin(np.pi / 2 - theta)],
                    [np.sin(np.pi / 2 - theta), np.cos(np.pi / 2 - theta)]]
                ).squeeze()

                # if 0 velocity
                if torch.all(avg_vel[reciever] == 0):
                    rot_mat = np.eye(2)
                
                # rotate sender and record
                rot_pos_sender = np.matmul(rot_mat, avg_pos[sender].T).T
                rot_pos_rec = np.matmul(rot_mat, avg_pos[reciever].T).T
                rel_pos = rot_pos_rec - rot_pos_sender
                neighbors.append((reciever, rel_pos))

                # making sure the geometry makes sense
                # rot_avg_vel = np.matmul(rot_mat, (avg_vel[reciever] / np.linalg.norm(avg_vel[reciever])).T).T
                # non_rot_avg_vel = (avg_vel[reciever] / np.linalg.norm(avg_vel[reciever]))
                # plt.plot([0., avg_pos[sender][:, 0].item()], [0., avg_pos[sender][:, 1].item()], label='sen_pos')
                # plt.plot([0., avg_pos[reciever][:, 0].item()], [0., avg_pos[reciever][:, 1].item()], label='rec_pos')
                # plt.plot([0., non_rot_avg_vel[:, 0].item()], [0.,non_rot_avg_vel[:, 1].item()], label='rec_vel')
                # plt.plot([0., rot_avg_vel[:, 0].item()], [0., rot_avg_vel[:, 1].item()], label='rot_rec_vel')
                # plt.plot([0., rot_pos_sender[:, 0].item()], [0., rot_pos_sender[:, 1].item()], label='rot_sen_pos')
                # plt.plot([0., rot_pos_rec[:, 0].item()], [0., rot_pos_rec[:, 1].item()], label='rot_rec_pos')
                # plt.plot([0., rel_pos[:, 0].item()], [0., rel_pos[:, 1].item()], label='rel_pos')
                # plt.legend()
                # plt.show()

    # plotting
    fig, axs = plt.subplots(1, 3)

    # plot how many incoming vs outgoing edges for each fish
    sends = [i[0] for i in sen_rec_info]
    recs = [i[1] for i in sen_rec_info]

    axs[0].hist(sends, edgecolor='black', color='#0B4F6C', bins=np.linspace(-0.5, 9.5, 11), density=True)
    axs[0].set_xlabel('Fish')
    axs[0].set_ylabel('Density')
    axs[0].set_title('Senders')
    axs[0].set_xlim([-1, 8])
    axs[0].set_xticks(np.linspace(0, 7, 8))

    axs[1].hist(recs, edgecolor='black', color='#FF9B71', bins=np.linspace(-0.5, 9.5, 11), density=True)
    axs[1].set_xlabel('Fish')
    axs[1].set_ylabel('Density')
    axs[1].set_title('Recievers')
    axs[1].set_xlim([-1, 8])
    axs[1].set_xticks(np.linspace(0, 7, 8))

    # plot histogram of len of edges
    axs[2].hist(len_of_edges, color='#CC0000', edgecolor='black', density=True)
    axs[2].set_xlabel('Distance (mm)')
    axs[2].set_ylabel('Density')
    axs[2].set_title('Avg edge distance')

    # Plot the exact same density plot BUT it's with graphs instead
    for j in range(num_fish):
        fig, axs = plt.subplots()
        neigh_dist = np.array([i[1].detach().numpy() for i in neighbors if i[0] == j])
        neigh_dist = neigh_dist.squeeze()

        im = axs.hist2d(neigh_dist[:, 0], neigh_dist[:, 1], bins=100, density=True, cmap="BuPu")
        cbar = axs.figure.colorbar(im[3], ax=axs)
        cbar.ax.set_ylabel("Density", rotation=-90, va="bottom")
        axs.set_ylim([-1, 1])
        axs.set_xlim([-1, 1])
        axs.set_title(f"Fish {j}")
        plt.tight_layout()

    plt.tight_layout()
    plt.show()


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

def acc_vs_error(checkpoint, test_list):
    """
    Plots average acceleration of test set vs mse position predictions
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

    # loop through datasets
    mse_and_acc = []
    for dataset in test_list:
        test = gu.chunk_data_for_NRI(dataset, size=config['timesteps'])
        for itr, graphs in enumerate(test):
            if itr == len(test) - 1: # ignore last time step
                pass

            else:
                # predict the edges for first 49 frames
                logits = encoder(torch.FloatTensor(graphs.x).contiguous(), rel_rec, rel_send)
                edges = gun.gumbel_softmax(logits, tau=config['temp'], hard=config['hard'])
                prob = gun.my_softmax(logits, -1)

                # predict the next n timesteps given the edges
                next_graph = test[itr+1].x
                output = decoder(torch.FloatTensor(next_graph).contiguous(), edges, rel_rec, rel_send, 49)

                # calculate avg acceleration and mse
                frame_rate = 120
                mse = torch.nn.functional.mse_loss(output[:, :, :, :2].squeeze(), next_graph[:, :, 1:, :2].squeeze()).detach().numpy()

                acc = []
                for i in range(num_fish):
                    avg_acc_x = np.mean(np.convolve([1, -1], next_graph[0, i, :, 2], 'valid') * frame_rate)
                    avg_acc_y = np.mean(np.convolve([1, -1], next_graph[0, i, :, 3], 'valid') * frame_rate)
                    acc.append(np.sqrt(avg_acc_x ** 2 + avg_acc_y ** 2))
                
                acc = np.mean(acc)
                mse_and_acc.append([mse, acc])

    # plotting
    mse_and_acc = np.array(mse_and_acc)
    sorted_ = np.argsort(mse_and_acc, axis=0)
    sorted_mse_and_acc = mse_and_acc[sorted_[:, 1]]

    fig, axs = plt.subplots()
    axs.scatter(sorted_mse_and_acc[:,1], sorted_mse_and_acc[:,0], alpha=0.3)
    axs.set_xlabel('magnitude of average acceleration')
    axs.set_ylabel('mse')
    axs.set_yscale('log')
    axs.set_title('Acceleration vs Error')
    axs.set_ylim([0, 1000])
    plt.show()


def evaluate(model, no_jumps=True):
    """
    Returns validation loss, test loss, and a rollout inference of the test set
    """
    # load model
    checkpoint = torch.load(model, map_location=torch.device('cpu'))

    # Plotting loss
    # fig, axs = plt.subplots()
    # axs.plot(range(len(checkpoint['train_loss'])), checkpoint['train_loss'])
    # axs.set_xlabel('epochs')
    # axs.set_ylabel('loss')
    # plt.show()

    datasets = checkpoint['datasets']
    config = checkpoint['config']

    # get data split
    if no_jumps:
        train_list = []
        val_list = []
        test_list = []
        for dataset in datasets:
            data_list = pkl.load(open(dataset, 'rb'))

            # split into train and test
            train, val, test = gu.train_val_test(data_list)

            # split train val test into chunks TODO: incorporate different chunk every epoch
            test_list += [test]

    else:
        data_list = []
        for dataset in datasets:
            data_list += pkl.load(open(dataset, 'rb'))

        train, val, test_list = gu.train_val_test(data_list)

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
        mse_over_time, edges = rollout_inference_nri(checkpoint, test_list)
        plot_mse_over_time(mse_over_time)
        plot_rollout_like_NRI(checkpoint, test_list)
        acc_vs_error(checkpoint, test_list)
        # plot_predicted_edges_over_time(checkpoint)

    elif checkpoint['model_type'] == 'mpnn':
        rollout_preds = rollout_inference_acc(model, test)
    elif checkpoint['model_type'] == 'egnn' or 'egnn_paper':
        rollout_preds = rollout_inference_pos(model, test, checkpoint['model_type'])

    # printing results
    # print('val loss:', val_loss)
    # print('test loss:', test_loss)
    # plot_rollout(rollout_preds, test_list)


if __name__ == '__main__':
    saved_models = glob.glob('results/saved_models/*')
    most_recent = max(saved_models, key=os.path.getctime)
    print(most_recent)
    evaluate(most_recent, no_jumps=True)
