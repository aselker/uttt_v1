import itertools
import numpy as np

# O = -1, blank = 0, X = 1


def victory_state(board):
    """Returns 0 for cat's game or incomplete, NaN for both win."""
    h_sums = np.sum(board, axis=0)
    v_sums = np.sum(board, axis=1)
    sums = np.concatenate(
        (h_sums, v_sums, [board[0, 0] + board[1, 1] + board[2, 2], board[2, 0] + board[1, 1] + board[0, 2]])
    )

    o_wins = -3 in sums
    x_wins = 3 in sums

    if x_wins and o_wins:
        return np.nan
    elif x_wins:
        return 1.0
    elif o_wins:
        return -1.0
    else:
        return 0.0


goodnesses = {}


def calc_goodness(board):
    key = board.tobytes()
    if key in goodnesses:
        return goodnesses[key]
    zeros = np.array(np.nonzero(np.logical_not(board))).T
    victory_state_ = victory_state(board)
    if victory_state_:
        goodness = victory_state_
    elif not len(zeros):
        goodness = 0
    else:
        next_goodnesses = 0
        for zero_slot in zeros:
            this_board = board.copy()
            this_board[tuple(zero_slot)] = -1
            next_goodnesses += calc_goodness(this_board)
            this_board[tuple(zero_slot)] = 1
            next_goodnesses += calc_goodness(this_board)
        goodness = next_goodnesses / (len(zeros) * 2)
    goodnesses[key] = goodness
    return goodness

if __name__ == "__main__":
    board = np.zeros((3,3), dtype=np.int8)
    board[0,0] = 1
    board[1,1] = 1
    board[2,2] = -1
    print(calc_goodness(board))

