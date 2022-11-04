import itertools
import numpy as np
import numba


@numba.njit
def _slow_victory_state(board):
    """
    Returns 0 for cat's game or both win, NaN for incomplete.

    Var names assume that it's x's turn, but args and return are agnostic, use -1 and 1.

    NaN is interpreted as neither own that square, but also nobody can play there.  Useful for feeding in cxc victory states.  Note
    that that's sort of inverted from what the input is.
    """
    x_wins = (
        (board[0, 0] == 1 and board[1, 0] == 1 and board[2, 0] == 1)
        or (board[0, 1] == 1 and board[1, 1] == 1 and board[2, 1] == 1)
        or (board[0, 2] == 1 and board[1, 2] == 1 and board[2, 2] == 1)
        or (board[0, 0] == 1 and board[0, 1] == 1 and board[0, 2] == 1)
        or (board[1, 0] == 1 and board[1, 1] == 1 and board[1, 2] == 1)
        or (board[2, 0] == 1 and board[2, 1] == 1 and board[2, 2] == 1)
        or (board[0, 0] == 1 and board[1, 1] == 1 and board[2, 2] == 1)
        or (board[2, 0] == 1 and board[1, 1] == 1 and board[0, 2] == 1)
    )

    o_wins = (
        (board[0, 0] == 2 and board[1, 0] == 2 and board[2, 0] == 2)
        or (board[0, 1] == 2 and board[1, 1] == 2 and board[2, 1] == 2)
        or (board[0, 2] == 2 and board[1, 2] == 2 and board[2, 2] == 2)
        or (board[0, 0] == 2 and board[0, 1] == 2 and board[0, 2] == 2)
        or (board[1, 0] == 2 and board[1, 1] == 2 and board[1, 2] == 2)
        or (board[2, 0] == 2 and board[2, 1] == 2 and board[2, 2] == 2)
        or (board[0, 0] == 2 and board[1, 1] == 2 and board[2, 2] == 2)
        or (board[2, 0] == 2 and board[1, 1] == 2 and board[0, 2] == 2)
    )

    if (not x_wins) and (not o_wins):
        if 0 in board:
            return 3
        return 0
    elif x_wins:
        return 1
    return 2


_victory_lut = np.empty((4,) * 9, dtype=np.int8)


def _lut_index_to_cxc(index):
    return np.reshape(index, (3, 3))


for index in itertools.product((0, 1, 2, 3), repeat=9):
    _victory_lut[index] = _slow_victory_state(_lut_index_to_cxc(index))


def victory_state(cxc_):
    return _victory_lut[tuple(np.concatenate(cxc_))]


goodnesses = {}


def calc_goodness(board):
    raise NotImplementedError("whyyy")
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
    tests = [
        (
            [
                [0, 0, 0],
                [1, 1, 1],
                [0, 0, 0],
            ],
            1,
        ),
        (
            [
                [2, 3, 1],
                [0, 2, 0],
                [1, 0, 2],
            ],
            2,
        ),
    ]

    for test in tests:
        vs = victory_state(test[0])
        if vs != test[1]:
            print("Failed:", test, "gives", vs)
