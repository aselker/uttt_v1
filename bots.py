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
            # return self.nn.predict(np.array([state_.ixi]), verbose=False)[0][0]
            return nn_common.call_model_on_states(self.nn, state_)

        super().__init__(value_function, "SimpleNnBot_" + filename)


class FasterSimpleNnBot:
    def __init__(self, filename):
        self.nn = nn_common.make_model()
        self.nn.load_weights(filename)
        self.name = "FasterSimpleNnBot_" + filename

    def get_move(self, state_):
        possible_moves = state_.list_valid_moves()
        possible_states = []
        for possible_move in possible_moves:
            possible_states.append(state_.copy())
            possible_states[-1].move(possible_move)
        values = nn_common.call_model_on_states(self.nn, possible_states)
        return possible_moves[np.argmin(values)]


class FasterMultiPlyNnBot:
    def __init__(self, filename, plies, deterministic=False):
        self.plies = plies
        self.nn = nn_common.make_model()
        self.nn.load_weights(filename)
        self.deterministic = deterministic
        self.name = "FasterMultiPlyNnBot_" + filename + "_" + str(plies)
        if deterministic:
            self.name = self.name + "_deterministic"

    def __str__(self):
        return self.name

    def get_move(self, state_):
        possible_moves = state_.list_valid_moves()
        possible_states = []
        for possible_move in possible_moves:
            possible_states.append(state_.copy())
            possible_states[-1].move(possible_move)
        values = self.get_values(possible_states, self.plies)
        if self.deterministic:
            return possible_moves[np.argmin(values)]
        else:
            values = np.array(values)
            if any(values == -1):  # If we can win, do.  Maybe this could be removed?
                return possible_moves[np.argmin(values)]
            if all(values == values[0]):  # Equal values, pick randomly.  Useful when all are value 1.
                return possible_moves[np.random.choice(np.arange(len(possible_moves)))]
            assert np.all(values > -1)  # Avoid 1/0
            weights = 1 / (1 + (values - 1) / 2) - 1
            assert np.all(0 <= weights)  # Negative weight?  Could just clip value I think, /shrug
            weights /= np.sum(weights)
            return possible_moves[np.random.choice(np.arange(len(possible_moves)), p=weights)]

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
            return [
                (0 if finished_victory_state == 2 else finished_victory_state)
                for finished_victory_state in finished_victory_states
            ]

        # Child values are positive if we leave the opponent in a good position.  First check their victory states; then, if
        # they're not finished, run the NN.
        # TODO: Skip this step if remaining_plies[0] > max(len(child_states_ragged))
        child_values_flat = np.array([state_.victory_state() for state_ in child_states_flat], dtype=float)
        unfinished = child_values_flat == 0
        child_values_flat[child_values_flat == 2] = 0
        if np.any(unfinished):  # Don't run on empty list, it makes Keras unhappy
            child_values_flat[unfinished] = nn_common.call_model_on_states(
                self.nn, np.array(child_states_flat, dtype=object)[unfinished]
            )

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
