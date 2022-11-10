import numpy as np


def swap_players(x):
    """Edits the passed-in array!"""
    if isinstance(x, int):
        if x == 2:
            return 1
        elif x == 1:
            return 2
        return x

    was_2 = x == 2
    x[x == 1] = 2
    x[was_2] = 1
    return x
