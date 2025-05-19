"""
Massive script for plotting a bunch of metrics for the fish data
"""

import numpy as np
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix


def order_parameter():
    """
    Order parameter over time
    """


def order_parameter_autocorr():
    """
    order parameter autocorrelation
    """


def plot_rotation(rel_pos, v, rel_pos_rot, rot_mat):
    norm_v = v / np.sqrt((v ** 2).sum())
    norm_v_rot = np.matmul(rot_mat, norm_v)

    fig, axs = plt.subplots()
    axs.scatter(rel_pos[:,0], rel_pos[:,1])
    axs.plot([0, norm_v[0]], [0, norm_v[1]], c='red')
    axs.scatter(rel_pos_rot[:,0], rel_pos_rot[:,1])
    axs.plot([0, norm_v_rot[0]], [0, norm_v_rot[1]], c='green')

    plt.show()


def nearest_neighbor_distribution(data_list):
    """
    Calculates a heat map for every fish the density of other fish around them
    """
    nnd = []

    # loop through different recordings
    for data in data_list:
        nnd_dataset = []

        # loop through time
        for t in data:
            rel_pos_list = []
            fish_pos = t.pos

            # loop through fish
            for p, v in zip(t.pos, t.vel):
                # find orientation
                theta = np.arctan(v[1] / v[0])
                if v[0] < 0:
                    theta += np.pi
                rot_mat = [[np.cos(np.pi / 2 - theta), -np.sin(np.pi / 2 - theta)],
                            [np.sin(np.pi / 2 - theta), np.cos(np.pi / 2 - theta)]]

                # calculate relative position
                rel_pos = fish_pos - p

                # orient fish
                if np.isnan(theta):
                    rel_pos_rot = rel_pos
                else:
                    rel_pos_rot = np.matmul(rot_mat, rel_pos.T).T

                # plot
                # plot_rotation(rel_pos, v, rel_pos_rot, rot_mat)

                rel_pos_list.append(rel_pos_rot)

            nnd_dataset.append(rel_pos_list)

        # append dataset
        nnd.append(nnd_dataset)

    # loop through every fish
    for i in nnd:
        tot_density = []

        # loop through number of fish
        for j in tqdm(range(len(nnd[0][0]))):
            fig, axs = plt.subplots()

            # getting all relative distances per fish
            points = np.array([snap[j] for snap in i])
            points = points.reshape((points.shape[0] * points.shape[1], points.shape[-1]))  # collapsing time
            points = np.delete(points, np.where(points == [0, 0])[0], axis=0)  # removing all the 0's

            # plotting 2d histogram
            im = axs.hist2d(points[:, 0], points[:, 1], bins=300, density=True, cmap="BuPu")
            cbar = axs.figure.colorbar(im[3], ax=axs)
            cbar.ax.set_ylabel("Density", rotation=-90, va="bottom")
            axs.set_ylim([-1, 1])
            axs.set_xlim([-1, 1])
            axs.set_title(f"Fish {j}")
            axs.set_xlim([-0.5, 0.5])
            axs.set_ylim([-0.5, 0.5])
            plt.tight_layout()

    plt.show()

        # plotting the total density of all fish combined TODO: takes way too long
        # tot_density = np.array(tot_density)
        # tot_density = tot_density.reshape((tot_density.shape[0] * tot_density.shape[1], tot_density.shape[-1]))

        # fig, axs = plt.subplots()
        # im = axs.hist2d(tot_density[:, 0], tot_density[:, 1], bins=20, density=True, cmap="RdPu")
        # cbar = axs.figure.colorbar(im[3], ax=axs)
        # cbar.ax.set_ylabel("Density", rotation=-90, va="bottom")
        # axs.set_ylim([-1, 1])
        # axs.set_xlim([-1, 1])
        # axs.set_title(f"All fish dataset {i}")


def vel_vel_corr(data):
    """
    calculates velocity-velocity auto-correlation values for each fish.
    Equationo taken from Vicsek and Zafeiris (2012)

    TODO: fix this / better understand what is happening
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


def directional_corr_fn(data):
    """
    A way to determine whether there is leader follower relationships given
    a time.
    """
    num_fish = data[0].pos.shape[0]
    vel_corr = []
    tau = np.arange(1000)  # NOTE: starting with 100 time lags
    for t in tqdm(tau):
        # get normal and offset values
        v_t = data[:-(t+1)]
        v_t_offset = data[t+1:]

        # calculate velocity outer product
        corr = []
        for i, j in zip(v_t, v_t_offset):
            norm_i = i.vel / np.expand_dims(np.linalg.norm(i.vel, axis=1), axis=1)
            norm_j = j.vel / np.expand_dims(np.linalg.norm(j.vel, axis=1), axis=1)

            # replacing nans
            norm_i = np.nan_to_num(norm_i)
            norm_j = np.nan_to_num(norm_j)
            corr.append(np.dot(norm_i, norm_j.T))

        corr = np.mean(corr, axis=0)
        vel_corr.append(corr)

    vel_corr = np.array(vel_corr)
    pkl.dump(vel_corr, open('results/vel_corr/vel_corr.pkl', 'wb'))

    # plotting autocorrelation
    fig, axs = plt.subplots()
    for i in range(num_fish):
        autocorr = [np.diag(j)[i] for j in vel_corr]
        axs.plot(tau, autocorr, label=f'fish {i+1}')

    axs.set_xlabel("Time lag ($\\tau$)")
    axs.set_ylabel('Correlation')
    plt.tight_layout()
    plt.show()

    # plotting velocity correlation between fish


if __name__ == '__main__':
    data = [pkl.load(open('data/fish/processed/8fish/240816f1.pkl', 'rb'))]
    nearest_neighbor_distribution(data)
    # data = pkl.load(open('data/fish/processed/8fish/240816f1.pkl', 'rb'))
    # directional_corr_fn(data)
