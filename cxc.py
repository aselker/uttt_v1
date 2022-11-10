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
        (board[0, 0] == -1 and board[1, 0] == -1 and board[2, 0] == -1)
        or (board[0, 1] == -1 and board[1, 1] == -1 and board[2, 1] == -1)
        or (board[0, 2] == -1 and board[1, 2] == -1 and board[2, 2] == -1)
        or (board[0, 0] == -1 and board[0, 1] == -1 and board[0, 2] == -1)
        or (board[1, 0] == -1 and board[1, 1] == -1 and board[1, 2] == -1)
        or (board[2, 0] == -1 and board[2, 1] == -1 and board[2, 2] == -1)
        or (board[0, 0] == -1 and board[1, 1] == -1 and board[2, 2] == -1)
        or (board[2, 0] == -1 and board[1, 1] == -1 and board[0, 2] == -1)
    )

    # Sort of awkward logic, to make the most-common case (nobody wins, game still going) the fastest.
    if (not x_wins) and (not o_wins):
        if 0 in board:
            return 0
        return 2
    elif x_wins and o_wins:
        return 2
    elif x_wins:
        return 1
    return -1


if __name__ == "__main__":
    tests = [
        (
            [
                [1, 0, 0],
                [1, -1, 1],
                [-1, 0, 0],
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
                [-1, 2, 1],
                [0, -1, 0],
                [1, 0, -1],
            ],
            -1,
        ),
        (
            [
                [-1, 2, 1],
                [1, 1, -1],
                [-1, -1, 1],
            ],
            2,
        ),
    ]

    tests = [(np.array(key), value) for (key, value) in tests]

    for test in tests:
        vs = victory_state(test[0])
        if vs != test[1]:
            print("Failed:", test, "gives", vs)
