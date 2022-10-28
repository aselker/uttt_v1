import numpy as np

import cxc


def pretty_print(ixi_):
    marks = {-1: "o", 0: " ", 1: "x"}
    vline = "|"
    hline = "-"
    cross = "+"
    grid = [[marks[s] for s in row] for row in np.hstack(np.hstack(ixi_))]

    for i, _ in enumerate(grid):
        for col in [3, 7]:
            grid[i].insert(col, vline)

    for i in [3,7]:
        grid.insert(i, hline * 3 + cross + hline * 3 + cross + hline * 3)

    return("\n".join(("".join(row) for row in grid)))

def victory_state(ixi_):
    cxc_states = np.array([[cxc.victory_state(cxc_) for cxc_ in row] for row in ixi_])
    return cxc.victory_state(cxc_states)


if __name__ == "__main__":
    starting_ixi = np.zeros((3, 3, 3, 3), dtype=np.int8)
