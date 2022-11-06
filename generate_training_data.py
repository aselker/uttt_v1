import sys
import pickle
import numpy as np
from copy import deepcopy

from mcts import Mcts
from state import State
import ixi

NUM_GAMES = 10
NUM_PLAYOUTS = 200
NUM_PREFILLED_EACH = 30


def main():
    assert len(sys.argv) == 2

    histories = [[] for _ in range(NUM_GAMES)]
    for game_index in range(NUM_GAMES):
        print("Game", game_index, "...")
        # Generate a game that's mostly full
        victory_state = 1
        while victory_state:
            to_sample = [1] * NUM_PREFILLED_EACH + [2] * NUM_PREFILLED_EACH + [0] * (81 - 2 * NUM_PREFILLED_EACH)
            to_sample = np.array(to_sample, dtype=np.uint8)
            if np.random.choice([0, 1]):
                to_sample[-1] = 2
            np.random.shuffle(to_sample)
            ixi_ = to_sample.reshape((3, 3, 3, 3))
            victory_state = ixi.victory_state(ixi_)

        possible_prev_moves = np.transpose((ixi_ == 2).nonzero())
        prev_move = tuple(possible_prev_moves[np.random.randint(len(possible_prev_moves))])
        state_ = State(ixi_, prev_move)
        mcts_ = Mcts(num_playouts=NUM_PLAYOUTS)

        while not victory_state:
            histories[game_index].append(deepcopy(state_))
            possible_moves = state_.list_valid_moves()
            values = []
            for possible_move in possible_moves:
                possible_state = deepcopy(state_)
                possible_state.move(possible_move)
                value = mcts_.get_value(possible_state)
                value = -value  # Because we want to leave the next player in the worst-possible state
                values.append(value)
            state_.move(possible_moves[np.argmax(values)])
            victory_state = state_.victory_state()

    with open(sys.argv[1], "wb") as f:
        pickle.dump(histories, f)


if __name__ == "__main__":
    main()
