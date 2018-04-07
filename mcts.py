"""
Implementation of Upper Confidence Trees MCTS for Othello
<David Kartchner>
<April 4, 2018>
# Python 3
"""

# Imports
import numpy as np
import numpy.random as rd
import datetime
from othello import *
import time
from tqdm import trange


# Helper Functions
def p_uct_score(w_i, n_i, n_p, c, p_i):
    """
    Calculate UCT score for a given node

    Parameters:
    ----------------
        w_i:    win count for given node
        n_i:    visit count for current node
        n_p:    visit count for parent node
        c:      exploration parameter (higher means more exploration)
        p_i:    policy score for a particular move

    Returns:
    ----------------
        u:      UCT score for given node
    """
    if n_i < 0 or n_p < 0:
        raise ValueError("Visit counts (n_i and n_p) must be positive!")
    return w_i/n_i + c*p_i*np.sqrt(np.log(n_p)/n_i)

def tuple_to_array(tup, shape=(8,8)):
    return np.array(list(tup)).reshape(shape)

def array_to_tuple(array):
    return tuple(array.flatten().tolist())


class MCTS(object):
    def __init__(self, p, board, player, CNN, c = 1):
        self.Q = 0
        self.N = 0
        self.wins = 0
        self.p = p
        self.board = board
        self.leaf = True
        self.player = player
        self.CNN = CNN
        self.c = c # Exploration constant
        # self.winner = False
        # self.loser = False

    def get_children(self):
        # Specify that node is no longer a leaf.
        self.leaf = False
        self.N += 1
        # Get legal moves
        xs, ys, boards = get_legal_moves(self.board, self.player)

        # Get CNN evaluation of moves
        nn_inputs = make_nn_inputs(self.board, self.player)
        self.move_eval = self.CNN.estimate_policy(nn_inputs)

        winner, end_game = check_game_over(board, player)
        if end_game:
            # Update note statistics
            self.wins += winner
            self.Q = self.wins / self.N

            # Send back the node's negative value to other player
            return self.player*winner
        else:
            d_wins = self.CNN.estimate_value(nn_inputs)
            self.wins += d_wins

        # This is the only move if we have to pass
        if len(xs) == 0: # Corresponds to passing
            move_p = self.move_eval[-1]
            self.children = [MCTS(move_p, self.board, -self.player, self.CNN, c=self.c)]
            self.move_positions = ["pass"]

        # Otherwise build a child for each possible move
        else:
            possible_moves = self.move_eval[:64].reshape(8,8)
            # Calc probabilities for each child
            self.child_ps = np.array([possible_moves[x, y] for x, y in zip(xs, ys)])
            self.children = [MCTS(self.child_ps[i], boards[i,:,:], -self.player, self.CNN, c=self.c) for i in range(len(xs))]
            # Keep track of move positions for each child
            self.move_positions = [(x,y) for x, y in zip(xs, ys)]

        # Update current node score
        self.Q = self.wins / self.N
        return self.player*d_wins

    def get_child_scores(self):
        scores = np.array([child.Q + self.c*child.p*(np.sqrt(self.N)/(child.N+1)) for child in self.children])
        return scores

    def build_tree(self):
        self.N += 1

        # If a leaf node, get the children and current valuation
        if self.leaf:
            d_wins = self.get_children()
            # print(self.board)

        # Otherwise, go further down the tree
        else:
            scores = self.get_child_scores()
            best = np.argmax(scores)
            d_wins = self.children[best].build_tree()

        # Update the current win count and node value
        self.wins += d_wins
        self.Q = self.wins/self.N

        # Return the negative of our d_wins to opponent
        return -d_wins

    def get_best_move(self, c=None):
        if c is not None:
            self.c = c
        scores = self.get_child_scores()
        best = np.argmax(scores)
        return self.children[best], self.move_positions(best)

    def get_move_probs(self):
        pass


class fakeCNN(object):
    def __init__(self):
        self.policy = np.random.rand(65)
        self.policy /= self.policy.sum()

    def estimate_policy(self, state):
        return self.policy

    def estimate_value(self, state):
        return np.random.rand()*2 - 1


if __name__ == '__main__':
    tic = time.time()
    game = Othello()
    board = game.board
    player = game.player
    tree_search = MCTS(1, board, player, fakeCNN())
    for i in trange(10000):
        tree_search.build_tree()
