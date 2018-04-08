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

    playing_net = ONET(n_filters=128, n_res=10)
    training_net = ONET(n_filters=128, n_res=10)
    # params to manually tune for above nets:
    #   size of game state storage in data manager
    #   scale for weight regularization
    #   number residual layers
    #   batch size and training iterations



    mcts = MCTS(1, board, game.player, playing_net,c=1)

    # Loop the following as long as we are playing games (1000 games or whatever):

    # set empty collections
    states = []
    pis = []
    zs = []

    # play 1 game:
    # presumably we'll follow some such structure:
    done = False
    while not done:

        # search available moves
        for i in trange(100):
            mcts.build_tree()
            # print("Child Qs:", np.array([child.Q for child in mcts.children]).flatten())
            # print("Child Ns:", np.array([child.N for child in mcts.children]).flatten())
            # print("Child Scores:",mcts.get_child_scores())
        print(mcts.get_move_probs())

        # save board state, save move probs (pi)
        states.append(game.get_game_state())
        pis.append(mcts.get_move_probs())

        # pick a move
        mcts.get_next_move()

        # make move

        # check if game is over
        # if game is over:
        #    done=True

    # zs += [winner for i in range(len(states)]

    # Having played one game, we can add the game data to the network being trained
    training_net.add_training_data(np.array(states), np.array(pis), np.array(zs)) # should check these are all right shape

    # End game playing loop.


    # Training network can now be trained.
    n_iters = 1000
    batch_size = 1024
    training_net.train(n_iters, batch_size, verbose=False)

    # play training_net against playing_net for 400 games.
    # If training_net beats playing_net 55% of the time, that is good.
    # Otherwise, that is not good.

    # if win_rate >= .55:
    #     copy weights from training_net into playing_net.  The following two lines should be good to go.
    #     training_net.save_weights()
    #     playing_net.restore_weights()

    # repeat all of the above from game playing loop.