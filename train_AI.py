import numpy as np
import tensorflow as tf
from othello import *
from mcts import MCTS
from alphello_net import Othello_Network as ONET
from tqdm import tqdm, trange




if __name__ == '__main__':
    game = Othello()
    board = np.abs(game.board)
    player = game.player
    cnn = ONET(n_filters=128, n_res=10)
    mcts = MCTS(1, board, game.player, cnn,c=1)
    for i in trange(100):
        mcts.build_tree()
        # print("Child Qs:", np.array([child.Q for child in mcts.children]).flatten())
        # print("Child Ns:", np.array([child.N for child in mcts.children]).flatten())
        # print("Child Scores:",mcts.get_child_scores())
    print(mcts.get_move_probs())
