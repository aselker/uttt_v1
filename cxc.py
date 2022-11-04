import itertools
import numpy as np
import numba


@numba.njit
def victory_state(board):
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
            return 0
        return 3
    elif x_wins:
        return 1
    return 2


if False:  # Cache victory states
    _victory_lut = np.empty((4,) * 9, dtype=np.int8)

    def _lut_index_to_cxc(index):
        return np.reshape(index, (3, 3))

    for index in itertools.product((0, 1, 2, 3), repeat=9):
        _victory_lut[index] = victory_state(_lut_index_to_cxc(index))

    def victory_state(cxc_):
        return _victory_lut[tuple(np.concatenate(cxc_))]


if __name__ == "__main__":
    tests = [
        (
            [
                [1, 0, 0],
                [1, 2, 1],
                [2, 0, 0],
            ],
            0,
        ),
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
        (
            [
                [2, 3, 1],
                [1, 1, 2],
                [2, 2, 1],
            ],
            3,
        ),
    ]

    for test in tests:
        vs = victory_state(test[0])
        if vs != test[1]:
            print("Failed:", test, "gives", vs)
