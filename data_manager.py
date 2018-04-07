import numpy as np


class Data_Manager():
    """
        Class for handling Othello state data
    """

    def __init__(self, states=None, pis=None, zs=None, max_size=50000):
        """
        :param states: np array of size N X board_size X board_size X 3, containing states
        :param pis: np array of size N X board_size**2 + 1, containing move probability distributions
        :param zs: np array of size N, containing winners of games
        :param max_size: maximum size of data store.
        """
        self.states = states
        self.pis = pis
        self.zs = zs

        self.data_ptr = 0
        self.max_size = max_size

    def add_data(self, s_, p_, z_):
        """
            Add new game data to store.  If new size, exceeds self.max_size, old stuff is overwritten.
            This ensures that the network only learns from the self.max_size most recent games.

        :param s_: np array of size N X board_size X board_size X 3, containing states
        :param p_: np array of size N X board_size**2 + 1, containing move probability distributions
        :param z_: np array of size N, containing winners of games
        """

        if self.states is None:
            self.states = s_
            self.pis = p_
            self.zs = z_
        else:
            self.states = np.vstack((self.states, s_))
            self.pis = np.vstack((self.pis, p_))
            self.zs = np.hstack((self.zs, z_))


        if len(self.states) > self.max_size:
            self.states = self.states[-self.max_size:]
            self.pis = self.pis[-self.max_size:]
            self.zs = self.zs[-self.max_size:]


    def get_batch(self, size=1024):
        """
            Return a mini-batch of data, sampled uniformly at random from data store.

        :param size: How many data items to fetch.
        :return: An array of states, array of move probs, array of winners.
        """
        if size > len(self.states): size = len(self.states)

        batch = np.random.choice(range(len(self.states)), size, replace=False)

        return self.states[batch], self.pis[batch], self.zs[batch]


