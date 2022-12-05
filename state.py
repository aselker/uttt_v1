import numpy as np
import itertools

import cxc
import ixi
import utils

# A move is a pair of (which cxc, move within cxc).  Each half is a pair of (horizontal index, vertical index).


class State:
    def __init__(self, ixi_=None, prev_move=None):
        self.ixi = ixi.empty if ixi_ is None else ixi_
        self.prev_move = prev_move

    def list_valid_moves(self):
        if ixi.victory_state(self.ixi):
            raise ValueError("Trying to list valid moves of a finished game")

        if self.prev_move is not None and not cxc.victory_state(self.ixi[self.prev_move[2:]]):
            valid_cxcs = [self.prev_move[2:]]
        else:
            valid_cxcs = [
                indices
                for indices in itertools.product((0, 1, 2), repeat=2)
                if not (cxc.victory_state(self.ixi[indices]))
            ]
        # At this point, valid_cxcs should be exactly the cxcs where it's legal to play, i.e. nobody has won and it's not a cat's game.
        valid_moves = []
        # This nested loop/list comp could probably be replaced with a numpy expression.
        for cxc_index in valid_cxcs:
            valid_moves += [
                cxc_index + space_index
                for space_index in itertools.product((0, 1, 2), repeat=2)
                if self.ixi[cxc_index + space_index] == 0
            ]
        return valid_moves

    def __hash__(self):
        return hash((ixi.hash(self.ixi), self.prev_move))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def victory_state(self):
        return ixi.victory_state(self.ixi)

    def move(self, move):
        assert self.ixi[move] == 0, "Invalid move"
        self.ixi[move] = 1
        self.prev_move = move
        # Switch whose turn it is: swap 1's and 2's.
        utils.swap_players(self.ixi)

    def __str__(self):
        return ixi.pretty_print(self.ixi, prev_move=self.prev_move)

    def __repr__(self):
        return self.__str__()

    def copy(self):
        return type(self)(ixi_=np.copy(self.ixi), prev_move=self.prev_move)
