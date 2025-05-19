"""
This is a script that, given data will animate how the fish over time.
"""

import numpy as np
import pickle as pkl
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def get_pos(data):
    """
    Gets fish position from pickled data
    """
    data = pkl.load(open(data, 'rb'))
    pos = np.array([i.pos for i in data])
    return pos


def down_sample(pos, frames=10000):
    """
    down sampling to a smaller amount of frames
    """
    chunks = pos.shape[0] // frames
    return pos[::chunks, :, :]


def make_animation(pos, edges=None, random_edges=False, boundary=1, filename="./fish_w_edges2.gif", same_color=False):
    """
    Makes animation out of the position data

    Args:
      edges - adjacecy matrix
    """
    # initial points
    fig, axs = plt.subplots()
    label_vals = np.arange(pos.shape[1])
    if same_color:
        scat = axs.scatter(pos[0, :, 0], pos[0, :, 1], c="#CC0000")
    else:    
        scat = axs.scatter(pos[0, :, 0], pos[0, :, 1], c=label_vals, cmap='tab10')

    # make edges
    if edges is None and random_edges is True:
        random_perm = np.random.permutation(len(pos[0,:,0]))
        edges_plot = axs.plot(pos[0, :, 0], pos[0, :, 1], c='gray', alpha=0.2)[0]
        edges_perm = axs.plot(pos[0, :, 0][random_perm], pos[0, :, 1][random_perm], c='gray', alpha=0.2)[0]
    
    elif edges is not None:
        print(edges)
        senders = pos[0, :, :][np.where(edges == 1)[0]]
        recievers = pos[0, :, :][np.where(edges == 1)[1]]
        edge_vals = np.concat([senders, recievers])
        edges_plot = axs.plot(edge_vals[:, 0], edge_vals[:, 1], c='gray', alpha=0.2)[0]

    else:
        pass

    scat.axes.set_ylim([-boundary, boundary])
    scat.axes.set_xlim([-boundary, boundary])

    def animate(i, edges, random_edges):
        scat.set_offsets(pos[i, :, :])

        if edges is None and random_edges is True:
            edges_plot.set_xdata(pos[i, :, 0])
            edges_plot.set_ydata(pos[i, :, 1])
            edges_perm.set_xdata(pos[i, :, 0][random_perm])
            edges_perm.set_ydata(pos[i, :, 1][random_perm])
            return (scat, edges, edges_perm)

        elif edges is not None:
            senders = pos[i, :, :][np.where(edges == 1)[0]]
            recievers = pos[i, :, :][np.where(edges == 1)[1]]
            edges = np.concat([senders, recievers])
            edges_plot.set_xdata(edges[:, 0])
            edges_plot.set_ydata(edges[:, 1])
            return (scat, edges)

        else:
            return scat

    ani = animation.FuncAnimation(
        fig=fig,
        func=partial(animate, edges=edges, random_edges=random_edges),
        frames=pos.shape[0],
        repeat=True
    )

    ani.save(filename=filename, writer="pillow", fps=120)


def main(data):
    # get positions
    pos = get_pos(data)

    # down sample data
    pos = down_sample(pos)

    # plot positions
    make_animation(pos)


if __name__ == '__main__':
    main('data/fish/processed/8fish/240816f1.pkl')
