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
from softmax import softmax


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
        self.Q = 0.
        self.N = 0.
        self.wins = 0. # NOTE: This is win from perspective of PARENT node
        self.p = p
        self.board = board
        self.leaf = True
        self.player = player
        self.CNN = CNN # NOTE: Neural network, on the other hand, evaluates how likely current node is to win
        self.c = c # Exploration constant
        # self.winner = False
        # self.loser = False

    def get_children(self):
        # Get legal moves
        xs, ys, boards = get_legal_moves(self.board, self.player)

        # Get CNN evaluation of moves
        nn_inputs = make_nn_inputs(self.board, self.player)
        self.move_eval = self.CNN.estimate_policy(nn_inputs).flatten()


        game_over, winner = check_game_over(self.board, self.player)
        if game_over:
            # Update note statistics
            d_wins = -self.player*winner
            self.wins += d_wins
            self.Q = self.wins / self.N
            # self.Q = 0

            # Send back the node's negative value to other player
            # return self.player*winner
            return d_wins

        else:
            self.leaf = False
            d_wins = -self.CNN.estimate_value(nn_inputs).flatten()
            self.wins += d_wins

        # This is the only move if we have to pass
        if len(xs) == 0: # Corresponds to passing
            move_p = self.move_eval[-1]
            self.children = [MCTS(1, self.board, -self.player, self.CNN, c=self.c)]
            self.move_positions = ["pass"]

        # Otherwise build a child for each possible move
        else:
            possible_moves = self.move_eval[:64].reshape((8,8))
            # Calc probabilities for each child
            self.child_ps = np.array([possible_moves[x, y] for x, y in zip(xs, ys)])
            self.child_ps /= self.child_ps.sum()
            self.children = [MCTS(self.child_ps[i], boards[i,:,:], -self.player, self.CNN, c=self.c) for i in range(len(xs))]
            # Keep track of move positions for each child
            self.move_positions = [(x,y) for x, y in zip(xs, ys)]

        # Update current node score
        self.Q = self.wins / self.N
        return d_wins

    def get_child_scores(self):
        scores = np.array([child.Q + self.c*child.p*(np.sqrt(self.N)/(child.N+1)) for child in self.children]).flatten()
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

    def get_next_move(self, tau=1, choose_best=False):
        """
        Choose next move in proportion to N^(1/tau), where N is the current
        visit count of each child
        """
        # First, get number of visits to each child
        visit_nums = np.array([child.N for child in self.children])

        # If in competitive play, simply choose the best most visited move
        if choose_best:
            best = np.argmax(visit_nums)
            return self.children[best], self.move_positions[best]

        # Otherwise, get visit probabilities for each node
        visit_nums = visit_nums**(1./tau)
        visit_probs = visit_nums/visit_nums.sum()
        next_node = np.random.choice(range(visit_probs.shape[0]), p=visit_probs)

        return self.children[next_node], self.move_positions[next_node]

    def get_move_probs(self, tau=1, return_child_probs=False):
        """
        Determine probability of making each move based on temperature tau
        """
        # Begin by getting visit count for each child
        visit_nums = np.array([child.N for child in self.children], dtype=np.float32)
        visit_logits = visit_nums**(1/tau)

        # Convert visit nums to move probabilities
        move_probs = visit_nums/visit_nums.sum()

        # If player has to pass, let this be the only move
        if self.move_positions[0] =="pass":
            all_move_probs = np.zeros(len(self.board.flatten())+1)
            all_move_probs[-1] = 1
            all_move_probs = all_move_probs.reshape((1,65))
            if return_child_probs:
                return all_move_probs, np.ones(1)
            return all_move_probs

        # Otherwise, label probabilities as they would be on the board
        board_move_probs = np.zeros(self.board.shape)
        for i, tup in enumerate(self.move_positions):
            x, y = tup
            board_move_probs[x,y] = move_probs[i]

        # Make move probs into the appropriately sized array and return
        all_move_probs = np.zeros(len(self.board.flatten())+1)
        all_move_probs[:-1] = board_move_probs.flatten()
        all_move_probs = all_move_probs.reshape((1,65))
        if return_child_probs:
            return  all_move_probs, move_probs
        return all_move_probs


    def play_move(self, n_iters=100):
        """
        Play a move against opponent neural Network
        """
        # Start with MCTS to get best move
        for i in xrange(n_iters):
            self.build_tree()
        child, move = self.get_next_move(choose_best=True)
        return child, move

    def receive_opponent_move(self, move):
        """
        Get move from opponent and update current state
        """
        child_ind = self.move_positions.index(move)
        return self.children[child_ind]

def selfplay(mcts_instance, search_iters=200):
    """
    Play self to generate CNN training data
    """
    # Set up game and get params
    board = mcts_instance.board
    boardsize = board.shape[0]
    player = mcts_instance.player

    # Set up arrays to keep track of players, game_states, move_probs
    states = np.zeros((100, boardsize, boardsize, 3))
    move_probs = np.zeros((100,65))

    # Keep track of whether game is over, as well as temperature on move choice
    game_over = False
    i = 0
    tau = 1

    while not game_over:
        # print(np.abs(mcts_instance.board).sum())
        # print("Player: {}, Board:".format(player))
        # print(mcts_instance.board)

        # Make tau small as game progresses
        if i == 20:
            tau = .1

        # Record current state
        states[i] = make_nn_inputs(board, player)

        # Check to see if game has ended
        game_over, winner = check_game_over(board, player)
        if game_over:
            outcome = np.ones(i-1) * winner
            final_states = states[:i-1,:,:,:]
            final_probs = move_probs[:i-1,:]
            break

        # If multiple moves, choose them via tree search
        for j in range(search_iters):
            mcts_instance.build_tree()

        # After performing tree search, pick a move and update
        move_probs[i], selection_probs = mcts_instance.get_move_probs(tau=tau, return_child_probs=True)
        move = np.random.choice(range(len(selection_probs)), p=selection_probs)
        mcts_instance = mcts_instance.children[move]
        player *= -1
        board = mcts_instance.board

        # Update iter count
        i += 1

    # print("Final Score: ", mcts_instance.board.sum())
    # print("Final Board: \n", mcts_instance.board)
    return final_states, final_probs, outcome

def play_game(black_net, white_net, mcts_iters=100):
    """
    Play a game between two neural networks to see which is better

    Inputs:
    ----------------------
        black_net - neural net to control black player
        white_net - neural net to control white player
        mcts_iters - number of iters to tree search for each move

    Returns:
    ----------------------
        winner - 1 if white, -1 of black
    """
    game = Othello()
    board = game.board

    # First define our black net, since it will go first.
    black_mcts = MCTS(1, board, -1, black_net)

    # Black begins by making a move, after which we use its board to
    # initialize white
    player = -1
    black_mcts, move = black_mcts.play_move(n_iters=mcts_iters)

    # Now initialize white once black has made its move.
    board = black_mcts.board
    white_mcts = MCTS(1, board, 1, white_net)
    player *= -1

    # Keep track of whether game is over
    game_over = False
    # Now actually play the game
    while not game_over:

        # Allow current player to tree_search for a move
        if player==-1:
            black_mcts, move = black_mcts.play_move(n_iters=mcts_iters)
            white_mcts = white_mcts.receive_opponent_move(move)
        else:
            white_mcts, move = white_mcts.play_move(n_iters=mcts_iters)
            black_mcts = black_mcts.receive_opponent_move(move)

        # Update current player and board and see if game is finished
        player *= -1
        board = black_mcts.board
        game_over, winner = check_game_over(board, player)
    return winner


def faceoff(new_net, old_net, matches=100, mcts_iters=100, tau = .1):
    """
    Make trained network play old network

    Inputs:
    ----------------------
        new_net - trained/updated network
    """
    game = Othello()
    black = game.player
    board = game.board

    winner_where_new_plays_black = []
    for _ in trange(50):
        pass


class fakeCNN(object):
    def __init__(self):
        self.policy = np.random.rand(65)
        self.policy /= self.policy.sum()

    def estimate_policy(self, state):
        return self.policy

    def estimate_value(self, state):
        return np.array([np.random.rand()*2 - 1])


if __name__ == '__main__':
    tic = time.time()
    game = Othello()
    board = game.board
    player = game.player
    # board = np.zeros((8,8))
    # board[2:-2, 2:-2] = 1
    # board[3:-3, 3:-3] = -1
    # player = 1
    overall_game_outcomes = []
    for i in trange(11):
        tree_search = MCTS(1, board, player, fakeCNN())
        final_states, final_probs, outcome = selfplay(tree_search, 200)
        overall_game_outcomes.append(outcome[-1])
    print(np.sum(overall_game_outcomes))
