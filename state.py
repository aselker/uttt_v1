import numpy as np
import itertools

import cxc
import ixi

# A move is a pair of (which cxc, move within cxc).  Each half is a pair of (horizontal index, vertical index).


class State:
    def __init__(self, ixi_=None, prev_move=None):
        self.ixi = ixi.empty if ixi_ is None else ixi_
        self.prev_move = prev_move

    def list_valid_moves(self):
        if self.prev_move is not None and np.isnan(cxc.victory_state(self.ixi[self.prev_move[1]])):
            valid_cxcs = [self.prev_move[1]]
        else:
            valid_cxcs = [indices for indices in itertools.product((0, 1, 2), repeat=2) if np.isnan(cxc.victory_state(self.ixi[indices]))]
        # At this point, valid_cxcs should be exactly the cxcs where it's legal to play, i.e. nobody has won and it's not a cat's game.
        valid_moves = []
        # This nested loop/list comp could probably be replaced with a numpy expression.
        for cxc_index in valid_cxcs:
            valid_moves += [
                cxc_index + space_index for space_index in itertools.product((0, 1, 2), repeat=2) if self.ixi[cxc_index + space_index] == 0
            ]
        return valid_moves


if __name__ == "__main__":
    # ixi_ = np.zeros((3, 3, 3, 3), dtype=np.int8)
    ixi_ = np.random.choice([-1, 0, 1], size=(3, 3, 3, 3))
    print(ixi.pretty_print(ixi_))
    state = State(ixi_=ixi_)
    valid_moves = state.list_valid_moves()
    print(len(valid_moves))
    print(valid_moves)
    breakpoint()
