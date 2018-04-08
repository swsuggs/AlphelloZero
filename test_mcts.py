import numpy as np
import tensorflow as tf
from othello import *
from mcts import MCTS
from alphello_net import Othello_Network as ONET
from tqdm import tqdm, trange


class fakeCNN(object):
    def __init__(self):
        self.policy = np.random.rand(65)
        self.policy /= self.policy.sum()

    def estimate_policy(self, state):
        return self.policy

    def estimate_value(self, state):
        return np.random.rand()*2 - 1

if __name__ == '__main__':
    game = Othello()
    board = np.abs(game.board)
    player = game.player
    print(check_game_over(board, player))

    playing_net = fakeCNN()
    mcts = MCTS(1, board, game.player, playing_net,c=1)
    for i in trange(100):
        mcts.build_tree()
