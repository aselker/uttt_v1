from copy import deepcopy
import numpy as np
import state
import random


class Mcts:
    def __init__(self, num_playouts=200):
        # Playouts are mappings of {game state : pair of (average playout outcome where -1 is guaranteed loss, number of playouts that have been done)}
        self.cache = {}
        self.num_playouts = num_playouts

    def run_playouts(self, state):
        if state not in self.cache:
            self.cache[state] = (0.0, 0)

        while self.cache[state][1] < self.num_playouts:
            self.run_playout(state)

    def update_cache(self, state, victory_state):
        if state not in self.cache:
            self.cache[state] = (0.0, 0)

        self.cache[state] = (
            (self.cache[state][0] * self.cache[state][1] + victory_state) / (self.cache[state][1] + 1),
            self.cache[state][1] + 1,
        )

    def run_playout(self, state):
        state = deepcopy(state)
        old_states = [deepcopy(state)]
        while np.isnan(state.victory_state()):
            state.move(random.choice(state.list_valid_moves()))
            old_states.append(deepcopy(state))  # This could keep hashes, which are smaller than objects, I guess

        victory_state = state.victory_state()
        assert victory_state in [0, -1]  # Can't win when your move is next
        for old_state in old_states[-1::-2]:  # States won by 1 (at the time)
            self.update_cache(old_state, victory_state)
        for old_state in old_states[-2::-2]:  # States lost by 1 (at the time)
            self.update_cache(old_state, -victory_state)
