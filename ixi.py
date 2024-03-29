import numpy as np
from colorama import Fore, Back, Style

import cxc

# Whose turn is this?
# The turn is 1's!
# The turn is 1's!
# It's mine, it's mine, it's mine
#
# By definition, it's always 1's turn next.


def pretty_print(ixi_, whose_turn="x", prev_move=None):
    assert False, "deprecated"
    if whose_turn == "x":
        marks = {0: " ", 1: "x", -1: "o", 2: "?"}
    elif whose_turn == "o":
        marks = {0: " ", 1: "o", -1: "x", 2: "?"}
    else:
        raise ValueError

    vline = "|"
    hline = "-"
    cross = "+"
    grid = [[marks[s] for s in row] for row in np.hstack(np.hstack(ixi_))]

    for i, _ in enumerate(grid):
        for col in [3, 7]:
            grid[i].insert(col, vline)

    # Color the most recent move
    if prev_move is not None:
        row = prev_move[0] * 3 + prev_move[2]
        col = prev_move[1] * 4 + prev_move[3]
        grid[row].insert(col, Fore.RED)
        grid[row].insert(col + 2, Fore.WHITE)

    for i in [3, 7]:
        grid.insert(i, hline * 3 + cross + hline * 3 + cross + hline * 3)

    return "\n" + Fore.WHITE + "\n".join(("".join(row) for row in grid))

def victory_state(ixi_):
    cxc_states = np.array([[cxc.victory_state(cxc_) for cxc_ in row] for row in ixi_])
    return cxc.victory_state(cxc_states)


def hash(ixi_):
    return ixi_.data.tobytes()


def empty():
    return np.zeros((3, 3, 3, 3), dtype=np.int8)


if __name__ == "__main__":
    # ixi_ = np.zeros((3, 3, 3, 3), dtype=np.int8)
    ixi_ = np.random.choice([0, 1, -1], size=(3, 3, 3, 3))
    print(pretty_print(ixi_))
    print(np.array([[cxc.victory_state(cxc_) for cxc_ in row] for row in ixi_]))
    print(victory_state(ixi_))
