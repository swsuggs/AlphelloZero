import numpy as np
import tensorflow as tf
from othello import *
from mcts import MCTS, selfplay, play_game, faceoff
from alphello_net import Othello_Network as ONET
from tqdm import tqdm, trange




if __name__ == '__main__':
    game = Othello()
    board = game.board
    player = game.player


    playing_net = ONET(n_filters=128, n_res=3)
    training_net = ONET(n_filters=128, n_res=3)

    training_net.load_weights()

    winners = []
    for k in trange(10):
        if k < 5:
            winners.append(-play_game(training_net, playing_net, mcts_iters=100))
        else:
            winners.append(play_game(playing_net, training_net, mcts_iters=100))

    # Calculate win rate
    win_rate = np.mean(winners)
    print(win_rate)
    print(winners)
