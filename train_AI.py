import numpy as np
import tensorflow as tf
from othello import *
from mcts import MCTS, selfplay, play_game, faceoff
from alphello_net import Othello_Network as ONET
from tqdm import tqdm, trange
import cPickle



if __name__ == '__main__':
    game = Othello()
    board = game.board
    player = game.player


    playing_net = ONET(n_filters=256, n_res=10)
    training_net = ONET(n_filters=256, n_res=10)
    # params to manually tune for above nets:
    #   size of game state storage in data manager
    #   scale for weight regularization
    #   number residual layers
    #   batch size and training iterations





    # Loop the following as long as we are playing games (1000 games or whatever):

    # set empty collections
    states = []
    pis = []
    zs = []

    # play 1 game:
    # presumably we'll follow some such structure:
    done = False
    iters = 0
    # while not done:
    new_net_wins = 0
    trained_net_win_rate = []
    for z in trange(1000):
        for _ in trange(100):
            tree_depth = 200
            training_mcts = MCTS(1, board, game.player, training_net,c=1)

            # Ger states, pis, and zs from a game of self-play
            states, pis, zs = selfplay(training_mcts,tree_depth)


            # Having played one game, we can add the game data to the network being trained
            training_net.add_training_data(np.array(states), np.array(pis), np.array(zs)) # should check these are all right shape

            # End game playing loop.


        # Training network can now be trained.
        n_iters = 10
        batch_size = 512
        training_net.train(n_iters, batch_size, verbose=False)


        # play training_net against playing_net for 100 games.
        # If training_net beats playing_net 55% of the time, that is good.
        # Otherwise, that is not good.
        winners = []
        for k in trange(10):
            if k < 5:
                winners.append(-play_game(training_net, playing_net, mcts_iters=tree_depth))
            else:
                winners.append(play_game(playing_net, training_net, mcts_iters=tree_depth))

        # Calculate win rate
        win_rate = np.mean(winners)
        print(win_rate)
        print(winners)
        trained_net_win_rate.append(win_rate)
        cPickle.dump(trained_net_win_rate, open("win_rates.cpkl",'wb'))

        if win_rate > .1 :
            # copy weights from training_net into playing_net.  The following two lines should be good to go.
            training_net.save_weights()
            playing_net.load_weights()
            new_net_wins += 1
            print("New Net win proportion", new_net_wins/(z+1))

        # if win_rate < -.1:
        #     playing_net.save_weights()
        #     training_net.load_weights()
        #     print("Old net is better")

    # repeat all of the above from game playing loop.
