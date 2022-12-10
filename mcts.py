import math
import numpy as np
import state
import utils
from nn_common import make_model


class Mcts:
    """Not actually Monte-Carlo tree search.  Still kinda works."""

    def __init__(self, num_playouts=2000):
        # Playouts are mappings of {game state : pair of (average playout outcome where -1 is guaranteed loss, number of playouts that have been done)}
        self.cache = {}
        self.num_playouts = num_playouts
        self.debug = False

    def get_value(self, state_):
        victory_state = state_.victory_state()
        if victory_state:
            return {2: 0, 1: 1, -1: -1}[victory_state]
        self.run_playouts(state_)
        if self.debug:
            print({self.reverse_cache[k]: v for k, v in self.cache.items()})
            breakpoint()
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


class ActualMcts(Mcts):
    """
    Maybe the cache should also store victory state and next valid states?
    """

    def __init__(self, num_playouts=2000, exploration_constant=np.sqrt(2)):
        super().__init__(num_playouts)
        self.exploration_constant = exploration_constant
        self.debug = False
        if self.debug:
            self.reverse_cache = {}

    def uct(self, state_, parent):
        # victory_state = state_.victory_state()
        # if victory_state == -1:
        #     return np.inf

        cache_entry = self.cache[hash(state_)]
        parent_entry = self.cache[hash(parent)]
        number_of_wins = (-cache_entry[0] + 1) / 2
        number_of_visits = cache_entry[1]
        parent_visits = parent_entry[1]

        return (number_of_wins / number_of_visits) + self.exploration_constant * math.sqrt(math.log(parent_visits) / number_of_visits)

    def run_playout(self, state_):
        state_ = state_.copy()
        if self.debug:
            self.reverse_cache[hash(state_)] = state_.copy()
        old_hashes = [hash(state_)]

        # Selection and expansion.
        while True:
            if state_.victory_state():
                old_hashes.append(hash(state_))
                if self.debug:
                    self.reverse_cache[hash(state_)] = state_.copy()
                break

            potential_states = []
            potential_state_has_been_visited = []
            for move in state_.list_valid_moves():
                potential_state = state_.copy()
                potential_state.move(move)
                potential_states.append(potential_state)
                potential_state_has_been_visited.append(hash(potential_state) in self.cache)

            if all(potential_state_has_been_visited):
                uct_scores = [self.uct(potential_state, state_) for potential_state in potential_states]
                state_ = potential_states[np.argmax(uct_scores)]
                old_hashes.append(hash(state_))
                if self.debug:
                    self.reverse_cache[hash(state_)] = state_.copy()
            else:
                unvisited_states = np.array(potential_states, dtype=object)[np.logical_not(potential_state_has_been_visited)]
                state_ = unvisited_states[np.random.randint(len(unvisited_states))]
                old_hashes.append(hash(state_))
                if self.debug:
                    self.reverse_cache[hash(state_)] = state_.copy()
                break

        if self.debug and False:
            # print({self.reverse_cache[k]: v for k,v in self.cache.items()})
            print({self.reverse_cache[k]: self.cache.get(k, None) for k in old_hashes})
            breakpoint()

        # Simulation.
        swap_parity = False
        victory_state = state_.victory_state()
        while not victory_state:
            valid_moves = state_.list_valid_moves()
            state_.move(valid_moves[np.random.randint(len(valid_moves))])
            swap_parity = not swap_parity
            victory_state = state_.victory_state()

        # Backpropagation.
        assert victory_state != 1  # Can't win when your move is next
        if swap_parity:
            victory_state = utils.swap_players(victory_state)
        for old_hash in old_hashes[-1::-2]:  # States won by 1 (at the time)
            self.update_cache(old_hash, victory_state)
        for old_hash in old_hashes[-2::-2]:  # States lost by 1 (at the time)
            self.update_cache(old_hash, utils.swap_players(victory_state))
