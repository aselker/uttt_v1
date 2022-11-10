import numpy as np
import state
import utils


class Mcts:
    def __init__(self, num_playouts=2000):
        # Playouts are mappings of {game state : pair of (average playout outcome where -1 is guaranteed loss, number of playouts that have been done)}
        self.cache = {}
        self.num_playouts = num_playouts

    def get_value(self, state_):
        victory_state = state_.victory_state()
        if victory_state:
            return victory_state
        self.run_playouts(state_)
        return self.cache[hash(state_)][0]

    def run_playouts(self, state_):
        hash_state = hash(state_)
        if hash_state not in self.cache:
            self.cache[hash_state] = (0.0, 0)

        while self.cache[hash_state][1] < self.num_playouts:
            self.run_playout(state_)

    def update_cache(self, hash_state, victory_state):
        if hash_state not in self.cache:
            self.cache[hash_state] = (0.0, 0)

        value_to_add = {2: 0, 1: 1, -1: -1}[victory_state]
        self.cache[hash_state] = (
            (self.cache[hash_state][0] * self.cache[hash_state][1] + value_to_add) / (self.cache[hash_state][1] + 1),
            self.cache[hash_state][1] + 1,
        )

    def run_playout(self, state_):
        state_ = state_.copy()
        old_hashes = [hash(state_)]
        victory_state = state_.victory_state()
        while not victory_state:
            valid_moves = state_.list_valid_moves()
            state_.move(valid_moves[np.random.randint(len(valid_moves))])
            old_hashes.append(hash(state_))
            victory_state = state_.victory_state()

        assert victory_state != 1  # Can't win when your move is next
        for old_hash in old_hashes[-1::-2]:  # States won by 1 (at the time)
            self.update_cache(old_hash, victory_state)
        for old_hash in old_hashes[-2::-2]:  # States lost by 1 (at the time)
            self.update_cache(old_hash, utils.swap_players(victory_state))
