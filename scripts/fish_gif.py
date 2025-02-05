"""
This is a script that, given data will animate how the fish over time.
"""

import numpy as np
import pickle as pkl
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


def make_animation(pos, boundary=150, filename="./fish.gif"):
    """
    Makes animation out of the position data
    """
    # initial points
    fig, axs = plt.subplots()
    label_vals = np.arange(pos.shape[1])
    scat = axs.scatter(pos[0, :, 0], pos[0, :, 1], c=label_vals, cmap='tab10')
    scat.axes.set_ylim([-boundary, boundary])
    scat.axes.set_xlim([-boundary, boundary])

    def animate(i):
        scat.set_offsets(pos[i, :, :])
        return scat

    ani = animation.FuncAnimation(
        fig=fig,
        func=animate,
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
