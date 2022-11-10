import numpy as np


def swap_players(x):
    """Edits the passed-in array!"""
    if isinstance(x, int):
        if x == -1:
            return 1
        elif x == 1:
            return -1
        return x

    was_1 = x == 1
    x[x == -1] = 1
    x[was_1] = -1
    return x
