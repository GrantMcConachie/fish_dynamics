"""
creating a bunch of different synthetic datasets
"""

import numpy as np
import matplotlib.pyplot as plt
from scripts.fish_gif import make_animation


class SpringSim():
    """
    Adapted from Kipf et. al. (2018)
    """
    def __init__(self, n_balls=5, box_size=5., loc_std=.5, vel_norm=.5,
                 interaction_strength=.1, noise_var=0.):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._spring_types = np.array([0., 0.5, 1.])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

    def _energy(self, loc, vel, edges):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] * (dist ** 2) / 2
            return U + K

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size  # if location outsize bounding box
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])  # reverse velocity

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def sample_trajectory(self, T=10000, sample_freq=10,
                          spring_prob=[1. / 2, 0, 1. / 2]):  # prev: [1. / 2, 0, 1. / 2]
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0

        # Sample edges (create a spring between points or not)
        edges = np.random.choice(self._spring_types,
                                 size=(self.n_balls, self.n_balls),
                                 p=spring_prob)
        edges = np.tril(edges) + np.tril(edges, -1).T  # makes it undirected
        np.fill_diagonal(edges, 0)  # no self edges

        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n) * self.loc_std
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            forces_size = - self.interaction_strength * edges
            np.fill_diagonal(forces_size,
                             0)  # self forces are zero (fixes division by zero)
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F

            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:  # only recording the sample freq
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                forces_size = - self.interaction_strength * edges
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n,
                                                                   n)))).sum(
                    axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F

            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            return loc, vel, edges


class LorenzSystem():
    def __init__(
            self,
            n_particles=8,
            sigma=10.,
            ro=28.,
            beta=8/3.,

    ):
        self.n_particles = n_particles
        self.sigma = sigma
        self.ro = ro
        self.beta = beta
        np.random.seed(42) # set random seed

    def _plot_lorenz(self):
        fig, axs = plt.subplots(1, (self.n_particles - 4))
        for i in range((self.n_particles-4)):
            axs[i].scatter(self.x[:, i], self.y[:, i])
        
        plt.tight_layout()
        plt.show()
    
    def sample_trajectory(self, time=100000, dt=1/120):
        """
        Generate a lorez system
        """
        # initialize values
        x_0 = np.random.randn(1, self.n_particles)
        y_0 = np.random.randn(1, self.n_particles)
        z_0 = np.random.randn(1, self.n_particles)

        # simulate values
        x = np.zeros((time + 1, self.n_particles))
        y = np.zeros((time + 1, self.n_particles))
        z = np.zeros((time + 1, self.n_particles))

        x[0] = x_0
        y[0] = y_0
        z[0] = z_0

        for t in range(time):
            x[t+1] = x[t] + dt * self.sigma * (y[t] - x[t])
            y[t+1] = y[t] + dt * (x[t] * (self.ro - z[t]) - y[t])
            z[t+1] = z[t] + dt * (x[t] * y[t] - self.beta * z[t])

        # for plotting
        self.x = x
        self.y = y
        self.z = z

        return np.stack([x, y, x], axis=-1)


if __name__ == '__main__':
    # Lorenz
    sim = LorenzSystem()
    sim.sample_trajectory()
    sim._plot_lorenz()

    # srping system
    # sim = SpringSim()
    # for i in range(5):
    #     loc, vel, edges = sim.sample_trajectory(T=100000, sample_freq=100)
    #     make_animation(
    #         np.einsum('ijk->ikj', loc),
    #         edges=None,
    #         filename=f"./results/figs/spring_mass/spring_{i}_no.gif",
    #         boundary=5,
    #         same_color=True
    #     )
    #     make_animation(
    #         np.einsum('ijk->ikj', loc),
    #         edges=edges,
    #         filename=f"./results/figs/spring_mass/spring_{i}_edges.gif",
    #         boundary=5,
    #         same_color=True
    #     )
    # print('done')
