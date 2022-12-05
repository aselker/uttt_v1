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
            if self.old_behavior: # XXX
                value = -value  # Because we want to leave the next player in the worst-possible state
            values.append(value)

        # Sort by ascending negative value -> best ones first
        values, possible_moves, possible_states = zip(*sorted(zip(values, possible_moves, possible_states), key=lambda p: -p[0]))

        next_values = []
        for possible_state in possible_states[: remaining_plies[0]]:
            next_values.append(self._value_function_helper(possible_state, remaining_plies[1:]))
        return -min(next_values)

    def __init__(self, filename, plies, old_behavior=False):
        self.plies = plies
        self.nn = nn_common.make_model()
        self.nn.load_weights(filename)
        self.old_behavior = old_behavior
        super().__init__(self.value_function, "MultiPlyNnBot_" + filename + "_" + str(plies) + str(old_behavior))


class FasterMultiPlyNnBot:
    def __init__(self, filename, plies):
        self.plies = plies
        self.nn = nn_common.make_model()
        self.nn.load_weights(filename)
        self.name = "FasterMultiPlyNnBot_" + filename

    def get_move(self, state_):
        possible_moves = state_.list_valid_moves()
        possible_states = []
        for possible_move in possible_moves:
            possible_states.append(state_.copy())
            possible_states[-1].move(possible_move)
        values = self.get_values(possible_states, self.plies)
        return possible_moves[np.argmin(values)]

    def value_function(self, state_):
        """For debug only"""
        return self.get_values([state_], self.plies)[0]

    def get_values(self, states, remaining_plies):
        # Child states are states that we could choose to leave the opponent in, by making some move.
        child_states_flat = []
        child_states_ragged = []
        finished_victory_states = np.zeros(len(states))
        for i, state_ in enumerate(states):
            if vs := state_.victory_state():
                finished_victory_states[i] = vs
                child_states_ragged.append([])
            else:
                possible_moves = state_.list_valid_moves()
                possible_states = []
                for possible_move in possible_moves:
                    possible_states.append(state_.copy())
                    possible_states[-1].move(possible_move)
                child_states_ragged.append(possible_states)
                child_states_flat += possible_states

        # If all child states are won or tied, don't bother with NN etc.
        if all(finished_victory_states):
            return [(0 if finished_victory_state == 2 else finished_victory_state) for finished_victory_state in finished_victory_states]

        # Child values are positive if we leave the opponent in a good position.
        # TODO: Check victory states of child states?  Or just trust the NN to notice them?
        # TODO: Skip this step if remaining_plies[0] > max(len(child_states_ragged))
        child_values_flat = nn_common.call_model_on_states(self.nn, child_states_flat)

        child_values_ragged = []
        for i, child_states in enumerate(child_states_ragged):
            child_values_ragged.append(child_values_flat[: len(child_states)])
            child_values_flat = child_values_flat[len(child_states) :]

        if remaining_plies == []:
            best_values_ragged = child_values_ragged
        else:
            # The recursive part.
            # Sort each state's children by value; make a flattened list of the best ones; run get_values on it;
            # find the best ones for each state; return those.

            best_states_ragged = []
            best_states_flat = []
            for child_states, child_values in zip(child_states_ragged, child_values_ragged):
                if not child_states:  # skip if was finished (victory_state != 0)
                    best_states_ragged.append([])
                else:
                    # Sort by ascending value, since we want to leave the opponent in the worst state.
                    _, sorted_states = zip(*sorted(zip(child_values, child_states), key=lambda p: p[0]))
                    best_states_ragged.append(sorted_states[: remaining_plies[0]])
                    best_states_flat += sorted_states[: remaining_plies[0]]

            # best_values is like child_values, so negative is good for us.
            best_values_flat = self.get_values(best_states_flat, remaining_plies[1:])

            best_values_ragged = []
            for i, best_states in enumerate(best_states_ragged):
                best_values_ragged.append(best_values_flat[: len(best_states)])
                best_values_flat = best_values_flat[len(best_states) :]

        values = []
        for finished_victory_state, best_values in zip(finished_victory_states, best_values_ragged):
            if finished_victory_state:
                # If we're passed a state that's been won by o, that means it's a bad state for us.  So don't invert.
                values.append(0 if finished_victory_state == 2 else finished_victory_state)
            else:
                values.append(-min(best_values))  # We can put them in the worst possible state

        if False:
            print("Best:", np.argmax(values), max(values))
            print(states[np.argmax(values)])
            print("Quick prediction:", nn_common.call_model_on_states(self.nn, states[np.argmax(values)]))
            print("Worst:", np.argmin(values), min(values))
            print(states[np.argmin(values)])
            print("Quick prediction:", nn_common.call_model_on_states(self.nn, states[np.argmin(values)]))

            import matplotlib.pyplot as plt

            quick_values = nn_common.call_model_on_states(self.nn, states)
            plt.plot(values, quick_values, ".")
            plt.show()

            breakpoint()

        return values


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
