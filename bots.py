import numpy as np
from mcts import Mcts
import nn_common


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
        self.nn = nn_common.make_model()
        self.nn.load_weights(filename)

        def value_function(state_):
            return self.nn.predict(np.array([state_.ixi]), verbose=False)[0][0]

        super().__init__(value_function, "SimpleNnBot_" + filename)


class MultiPlyNnBot(ValueFunctionBot):
    """
    TODO: Make this less loopy, but instead look at all of ply N at the same time.  More efficent.
    """

    def value_function(self, state_):
        return self._value_function_helper(state_, self.plies)

    def _value_function_helper(self, state_, remaining_plies):
        if victory_state := state_.victory_state():
            return 0 if victory_state == 2 else victory_state

        if len(remaining_plies) == 0:
            return nn_common.call_model_on_states(self.nn, [state_])[0]

        possible_moves = state_.list_valid_moves()
        values = []
        possible_states = []
        for possible_move in possible_moves:
            possible_state = state_.copy()
            possible_state.move(possible_move)
            possible_states.append(possible_state)
            value = nn_common.call_model_on_states(self.nn, [possible_state])[0]
            value = -value  # Because we want to leave the next player in the worst-possible state
            values.append(value)

        values, possible_moves, possible_states = zip(*sorted(zip(values, possible_moves, possible_states), key=lambda p: -p[0]))

        next_values = []
        for possible_state in possible_states[: remaining_plies[0]]:
            next_values.append(self._value_function_helper(possible_state, remaining_plies[1:]))
        return -min(next_values)

    def __init__(self, filename, plies):
        self.plies = plies
        self.nn = nn_common.make_model()
        self.nn.load_weights(filename)
        super().__init__(self.value_function, "MultiPlyNnBot_" + filename + "_" + str(plies))


class FasterMultiPlyNnBot:
    def get_move(self, state_):
        possible_moves = state_.list_valid_moves()
        possible_states = [state_.copy().move(possible_move) for possible_move in possible_moves]
        values = self.get_values(possible_states)
        return possible_moves[np.argmax(values)]

    def get_values(self, states, remaining_plies):
        """Like get_value, but takes and returns multiple states."""
        all_possible_states
        for state_ in states:
            possible_moves = state_.list_valid_moves()
            possible_states = [state_.copy().move(possible_move) for possible_move in possible_moves]
            all_possible_states += possible_states

        this_ply_values = self.nn.predict(np.array([all_possible_state.ixi]), verbose=False)[0][0]


class RandomBot:
    def __init__(self):
        self.name = "RandomBot"

    def get_move(self, state_):
        possible_moves = state_.list_valid_moves()
        return possible_moves[np.random.choice(range(len(possible_moves)))]


class HumanBot:
    def __init__(self):
        self.name = "HumanBot"

    def get_move(self, state_):
        print(state_)
        while True:
            human_input = input()
            if len(human_input) != 4:
                print("format: 0000 to 2222")
                continue
            try:
                move = tuple([int(h) for h in human_input])
            except ValueError:
                print("format: 0000 to 3333")
                continue
            if move not in state_.list_valid_moves():
                print("invalid move")
                continue
            return move
