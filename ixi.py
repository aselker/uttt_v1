import numpy as np

import cxc

# Whose turn is this?
# The turn is 1's!
# The turn is 1's!
# It's mine, it's mine, it's mine
#
# By definition, it's always 1's turn next.


def pretty_print(ixi_, whose_turn="x"):
    match whose_turn:
        case "x":
            marks = {-1: "o", 0: " ", 1: "x"}
        case "o":
            marks = {1: "o", 0: " ", -1: "x"}
        case _:
            raise ValueError

    vline = "|"
    hline = "-"
    cross = "+"
    grid = [[marks[s] for s in row] for row in np.hstack(np.hstack(ixi_))]

    for i, _ in enumerate(grid):
        for col in [3, 7]:
            grid[i].insert(col, vline)

    for i in [3, 7]:
        grid.insert(i, hline * 3 + cross + hline * 3 + cross + hline * 3)

    return "\n".join(("".join(row) for row in grid))


def victory_state(ixi_):
    # TODO: Check this.  cxc.victory_state isn't geared to deal with cells where nobody owns it and also nobody can play there.
    cxc_states = np.array([[cxc.victory_state(cxc_) for cxc_ in row] for row in ixi_])
    cxc_states[np.isnan(cxc_states)], cxc_states[not cxc_states] = 0, np.nan
    return cxc.victory_state(cxc_states)


def hash(ixi_):
    return ixi.data.tobytes()


def empty():
    return np.zeros((3, 3, 3, 3), dtype=np.int8)
