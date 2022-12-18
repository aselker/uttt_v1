import numpy as np
import itertools
from colorama import Fore, Back, Style

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
            valid_cxcs = [indices for indices in itertools.product((0, 1, 2), repeat=2) if not (cxc.victory_state(self.ixi[indices]))]
        # At this point, valid_cxcs should be exactly the cxcs where it's legal to play, i.e. nobody has won and it's not a cat's game.
        valid_moves = []
        # This nested loop/list comp could probably be replaced with a numpy expression.
        for cxc_index in valid_cxcs:
            valid_moves += [
                cxc_index + space_index for space_index in itertools.product((0, 1, 2), repeat=2) if self.ixi[cxc_index + space_index] == 0
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
        return self.pretty_print()

    def pretty_print(self, whose_turn="x"):
        if whose_turn == "x":
            marks = {0: " ", 1: "x", -1: "o", 2: "?"}
        elif whose_turn == "o":
            marks = {0: " ", 1: "o", -1: "x", 2: "?"}
        else:
            raise ValueError

        vline = "|"
        hline = "-"
        cross = "+"

        slot_strings = []
        for i in range(3):
            slot_strings.append([])
            for j in range(3):
                slot_strings[-1].append([])
                for k in range(3):
                    slot_strings[-1][-1].append([])
                    for l in range(3):
                        slot_strings[-1][-1][-1].append(marks[self.ixi[i, j, k, l]])
        slot_strings = np.array(slot_strings, dtype="<U64")

        if self.prev_move is not None:
            slot_strings[self.prev_move] = Fore.RED + slot_strings[self.prev_move] + Fore.WHITE

        for move in self.list_valid_moves():
            slot_strings[move] = Back.BLUE + slot_strings[move] + Back.RESET

        for i in range(3):
            for j in range(3):
                vs = cxc.victory_state(self.ixi[i, j])
                color = {
                    -1: Fore.LIGHTRED_EX,
                    0: Fore.RESET,
                    1: Fore.LIGHTBLUE_EX,
                    2: Fore.WHITE,
                }[vs]
                for k in range(3):
                    for l in range(3):
                        slot_strings[i, j, k, l] = color + slot_strings[i, j, k, l] + Fore.RESET

        grid = np.full((11, 17), " ", dtype="<U64")  # Thanks I hate prime numbers
        for i in range(3):
            grid[4 * i : 4 * (i + 1) - 1, ::2] = np.concatenate(slot_strings[i].transpose(0, 2, 1), axis=0).T

        grid[:, [5, 11]] = vline
        grid[[3, 7], :] = hline
        grid[(3, 3, 7, 7), (5, 11, 5, 11)] = cross

        return "\n" + Fore.WHITE + "\n".join(("".join(row) for row in grid))

    def __repr__(self):
        return self.__str__()

    def copy(self):
        return type(self)(ixi_=np.copy(self.ixi), prev_move=self.prev_move)
