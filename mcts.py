import numpy as np
import state
import utils
from nn_common import make_model


class Mcts:
    def __init__(self, num_playouts=2000):
        # Playouts are mappings of {game state : pair of (average playout outcome where -1 is guaranteed loss, number of playouts that have been done)}
        self.cache = {}
        self.num_playouts = num_playouts

    def get_value(self, state_):
        victory_state = state_.victory_state()
        if victory_state:
            return {2: 0, 1: 1, -1: -1}[victory_state]
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


class ValueFunctionBot:
    def __init__(self, value_function, name):
        self.value_function = value_function
        self.name = name

    def get_move(self, state_):
        possible_moves = state_.list_valid_moves()
        values = []
        for possible_move in possible_moves:
            possible_state = state_.copy()
            possible_state.move(possible_move)
            value = self.value_function(possible_state)
            value = -value  # Because we want to leave the next player in the worst-possible state
            values.append(value)
        return possible_moves[np.argmax(values)]


class MctsBot(ValueFunctionBot):
    def __init__(self, num_playouts=100):
        self.mcts = Mcts(num_playouts=num_playouts)
        name = "MctsBot" + str(num_playouts)
        super().__init__(self.mcts.get_value, name)


class SimpleNnBot(ValueFunctionBot):
    def __init__(self, filename):
        self.nn = make_model()
        self.nn.load_weights(filename)

        def value_function(state_):
            return self.nn.predict(state_.ixi, verbose=False)[0]

        super().__init__(value_function, "SimpleNnBot_" + filename)


class RandomBot:
    def __init__(self):
        self.name = "RandomBot"

    def get_move(self, state_):
        possible_moves = state_.list_valid_moves()
        return possible_moves[np.random.choice(range(len(possible_moves)))]
