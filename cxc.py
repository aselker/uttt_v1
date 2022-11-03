import itertools
import numpy as np


def victory_state(board):
    """
    Returns 0 for cat's game or both win, NaN for incomplete.

    Var names assume that it's x's turn, but args and return are agnostic, use -1 and 1.

    NaN is interpreted as neither own that square, but also nobody can play there.  Useful for feeding in cxc victory states.  Note
    that that's sort of inverted from what the input is.
    """
    h_sums = np.sum(board, axis=0)
    v_sums = np.sum(board, axis=1)
    sums = np.concatenate((h_sums, v_sums, [board[0, 0] + board[1, 1] + board[2, 2], board[2, 0] + board[1, 1] + board[0, 2]]))

    o_wins = -3 in sums
    x_wins = 3 in sums

    if x_wins and o_wins:
        return 0.0
    elif x_wins:
        return 1.0
    elif o_wins:
        return -1.0
    elif 0 in board:
        return np.nan
    else:
        return 0.0


goodnesses = {}


def calc_goodness(board):
    raise ThisIsProbablyOutOfDateError
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
    board = np.zeros((3, 3), dtype=np.int8)
    board[0, 0] = 1
    board[1, 1] = 1
    board[2, 2] = -1
    print(calc_goodness(board))
