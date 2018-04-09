"""
Othello game simulation
<David Kartchner>
<3/18/2018>
"""


import numpy as np
from tqdm import tqdm
from string import ascii_lowercase

def get_shifts(arr):
    """
    Shift array 1 element right, left, up, down, and diagnoal. Total 8 shifts.
    """
    # Get dimensions of input array
    m,n = arr.shape

    # Create 0 array with row corresponding to each shift
    shifts = np.zeros((8,m,n))

    # Calculate lateral shifts
    shifts[0,:,1:] = arr[:,:-1]
    shifts[1,:,:-1] = arr[:,1:]

    # Calculate vertical shifts
    shifts[2,1:,:] = arr[:-1,:]
    shifts[3,:-1,:] = arr[1:,:]

    # Calculate diagnoal shifts
    shifts[4,1:,1:] = arr[:-1,:-1]
    shifts[5,1:,:-1] = arr[:-1,1:]
    shifts[6,:-1,1:] = arr[1:,:-1]
    shifts[7,:-1,:-1] = arr[1:,1:]

    return shifts


def check_row(input_row, pos, piece_to_place):
    """
    Check is a move is legal in a given row

    Inputs:
        row - numpy array representing row to check
                (could be vertical or diag of board also)
        pos - position in row where you expect to put piece
        piece_to_place - what piece to play in row (should be in {-1,1})

    Returns:
        is_valid - bool representing whether move is valid in row
        newrow - row with flipped pieces (if any)
    """
    # print("piece to place: {}".format(piece_to_place))
    row = input_row.copy()
    # Make position is valid
    if row[pos] != 0:
        raise ValueError("Can only play piece on empty square!")

    # Check if row is too short
    n = len(row)
    valid_move = 0
    if n < 3:
        return 0, row

    # First check if piece is at endpoints
    # Left endpoint
    if pos == 0:
        if (row[pos+1] == piece_to_place) or (row[pos+1] == 0):
            return 0, row
        else:
            for i in range(pos+2, n):
                if row[i]==0:
                    return 0, row
                if row[i] == piece_to_place:
                    row[:i] = piece_to_place
                    return 1, row
            return 0, row

    # Now check right endpoint
    elif pos == n - 1 :
        if row[pos-1] == piece_to_place or row[pos-1] == 0:
            return 0, row
        else:
            for i in range(pos-2, -1, -1):
                if row[i]==0:
                    return 0, row
                if row[i] == piece_to_place:
                    row[i:] = piece_to_place
                    return 1, row

            return 0, row

    # Now look at cases where it might be in the middle
    else:
        if (row[pos-1] in set([0,piece_to_place])) & (row[pos+1] in set([0, piece_to_place])):
            return 0, row

        # Account for cases where we play next to end piece
        if pos+1 < n:
            # Otherwise, loop through until we find one of our pieces
            for i in range(pos+1, n):
                # Empty space
                if row[i]==0:
                    break
                # Flanked by our pieces
                if row[i]==piece_to_place:
                    valid_move = 1
                    row[pos:i] = piece_to_place

        # Case where we play next to front spot
        if pos-1 >= 0:
            # Otherwise, find one of our pieces or an empty space
            for i in range(pos-1, -1, -1):
                if row[i]==0:
                    break
                if row[i]==piece_to_place:
                    valid_move = 1
                    row[i:pos+1] = piece_to_place

        return valid_move, row


def check_legal(board, player, row, col, boardsize=8):
    """
    Check if a placing a piece at a particular position is legal
    """
    n = boardsize-1
    # keep track of what positions have legal moves
    move_legality = np.zeros(4)
    # Begin by checking diagnonals
    # Top-left to bottom-right
    k1 = col - row
    diag1 = np.diag(board, k1)
    if k1 >=0:
        move_legality[0], new_diag1 = check_row(diag1, row, player)
    else:
        move_legality[0], new_diag1 = check_row(diag1, col, player)

    # Bottom-left to top-right
    k2 = n - col - row
    diag2 = np.diag(np.rot90(board, k=-1), k2)
    if k2 >= 0 :
        move_legality[1], new_diag2 = check_row(diag2, col, player)
    else:
        move_legality[1], new_diag2 = check_row(diag2, n-row, player)

    # Row
    move_legality[2], new_row = check_row(board[row,:], col, player)

    # Column
    move_legality[3], new_col = check_row(board[:, col], row, player)
    is_legal = int(move_legality.sum()>0)

    # If move is legal, make new board that would be created
    if is_legal:
        new_board = np.copy(board)

        # Put in main diagonal
        for i in range(1,diag1.shape[0]):
            if k1 >= 0:
                new_board[i, k1+i] = new_diag1[i]
            else:
                new_board[i-k1, i] = new_diag1[i]

        # Put other diagonal
        for j in range(1,diag2.shape[0]):
            if k2 >= 0:
                new_board[n-k2-j, j] = new_diag2[j]
            else:
                new_board[n-j, j-k2] = new_diag2[j]
        # Put in new rows/columns
        new_board[row,:] = new_row
        new_board[:,col] = new_col
        new_board[row, col] = player
        return 1, new_board
    else:
        return 0, 0


def adj_to_opponent(board, player):
    """
    Get list of spaces adjacent to opponent pieces that are empty
    """
    opponent_pieces = (board == -player).astype(np.int8)
    adj_open_spaces = (get_shifts(opponent_pieces).sum(axis=0) > 0)
    adj_open_spaces[board !=0] = 0
    return adj_open_spaces.astype(np.uint8)




def get_legal_moves(board, player, boardsize=8):
    """
    Calculate possible that a certain player can take
    """
    # Get spaces adjacent to opponent's current pieces
    adj_to_opp =adj_to_opponent(board, player)
    possible_x, possible_y = np.where(adj_to_opp==1)
    legal_moves = np.zeros(possible_x.shape[0])
    possible_boards = np.zeros((possible_x.shape[0], boardsize, boardsize))

    for i in range(possible_x.shape[0]):
        x, y = possible_x[i], possible_y[i]
        legal_moves[i], possible_boards[i] = check_legal(board, player, x,y)

    # Return x and y coords of legal moves, along with new boards generated by each
    possible_x = possible_x[legal_moves==1]
    possible_y = possible_y[legal_moves==1]
    possible_boards = possible_boards[legal_moves==1]
    return possible_x, possible_y, possible_boards


def check_game_over(board, player, boardsize=8):
    if np.abs(board).sum() == boardsize**2:
        return True, np.sign(board.sum())
    if len(get_legal_moves(board, player)[0]) == 0:
        player *= -1
        if len(get_legal_moves(board, player)[0]) == 0:
            return True, np.sign(board.sum())
    return False, None


def make_nn_inputs(board, player):
    """
    Format board to be passed to polity/value evaluation neural network
    """
    n = board.shape[0]
    nn_inputs = np.ones((n,n,3))
    nn_inputs[:,:,0] *= (board == 1)
    nn_inputs[:,:,1] *= (board == -1)
    nn_inputs[:,:,2] *= player
    return nn_inputs.reshape((1,n,n,3))




class Othello(object):
    """
    Class to represent a game of Othello
    """

    def __init__(self,
                 board=None,
                 boardsize=(8,8),
                 player=-1,
                 black="AI",
                 white="AI"):
        if board is not None:
            boardsize = board.shape
        # Initialize board
        # Make sure board dimensions satisfy constraints
        n = boardsize[0]
        m = boardsize[1]
        if m != n:
            raise ValueError("Board must be square")

        if n%2 != 0 or n < 0:
            raise ValueError("boardsize values must be positive even integers")

        self.boardsize = n

        if board is None:
            # Make board as numpy array
            # White pieces are represented by 1, black by -1
            self.board = np.zeros(boardsize, dtype=np.int8)

            # Place pieces
            self.board[(n//2 -1):(n//2 +1), (n//2-1):(n//2)+1] = 1
            self.board[n//2 -1, n//2] *= -1
            self.board[n//2, n//2 -1] *= -1
        else:
            self.board = board

        # Keep track of whose turn it is
        # Black always goes first
        self.player = player

        # Make dict of board positions to make it easier for human players
        self._lettermap = {letter:i for i, letter in enumerate(ascii_lowercase)}


    def get_game_state(self):
        state = np.zeros((self.boardsize, self.boardsize, 3))  # this is only working for a state that includes no history, only current
        state[:,:,0][self.board<0] = -1
        state[:,:,1][self.board>0] = 1
        state[:,:,2] = self.player
        return state


    def _calc_current_score(self):
        """
        Get the relative score of the board (positive means white is winning)
        """
        return self.board.sum()



    def _play_move(self, position):
        """
        Place a piece and update board and current player.

        Inputs:
        --------------------
            position - a tuple containing the x and y coordinate

        Outputs:
        --------------------
            Updates internal board and player representations
        """
        x, y, boards = self.get_legal_moves(self.board, self.player)
        move_updates = {(x[i],y[i]):boards[i] for i in range(x.shape[0])}
        legal_moves = set(move_updates.keys())
        if position not in legal_moves:
            raise ValueError("Move is not legal!")
        self.board = move_updates[position]
        self.player *= -1





    def get_winner(self):
        full_board =  np.abs(self.board).sum() == self.boardsize**2

    def play_game(self, human_black=False, human_white=False):
        black_pass=False
        white_pass=False
        for play in range(60):
            pass

if __name__ == '__main__':
    # full_test_board = np.zeros((8,8))
    # test_array = np.array([[ 1, 1, 1, 1],
    #                        [-1, 1,-1, 1],
    #                        [ 1, 0, 1,-1],
    #                        [-1, 1, 1,-1]])
    # full_test_board[2:6, 2:6] = test_array
    # game = Othello(full_test_board.astype(np.int8))


    # Test legal move checker
    # game = Othello()
    # print(game.board)
    # print(adj_to_opponent(game.board, game.player))

    # print(get_legal_moves(game.board, game.player))
    # game._play_move((2,3))
    # print(game.board)
    # print(get_shifts(game.board))
    # print((get_shifts(game.board)*game.board))
    # print(game.get_legal_moves(self.board, self.player))

    # # Test cases for check_row
    # row1 = np.array([0,0,0,1,1,-1])
    # row2 = np.array([1,1,1,1,0,0])
    # row3 = np.array([1,1,0,1,1,1])
    row4 = np.array([-1,1,1,1,1,1,1,0])
    # row5 = np.array([1,-1,1,-1,1,0])
    # row6 = np.array([-1,1,0,1,-1,1])
    # row7 = np.array([1,0])
    # row8 = np.array([1,0,1])
    # row9 = np.array([0,0,1,-1,0,0])
    # print(row1, check_row(row1, 2,-1))
    # print(row1, check_row(row1, 2, 1))
    # print(row2, check_row(row2, 4,-1))
    # print(row2, check_row(row2, 4, 1))
    # print(row3,check_row(row3, 2,-1))
    # print(row3,check_row(row3, 2, 1))
    print(row4,check_row(row4, 7,-1))
    print(row4,check_row(row4, 7, 1))
    # print(row5,check_row(row5, 5,-1))
    # print(row5,check_row(row5, 5, 1))
    # print(row6,check_row(row6, 2,-1))
    # print(row6,check_row(row6, 2, 1))
    # print(row7,check_row(row7, 1,-1))
    # print(row7,check_row(row7, 1, 1))
    # print(row8,check_row(row8, 1,-1))
    # print(row8,check_row(row8, 1, 1))
    # print(row9,check_row(row9, 4,-1))
    # print(row9,check_row(row9, 4, 1))
