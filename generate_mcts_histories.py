import sys
import pickle
import numpy as np
import itertools

from mcts import Mcts
from state import State
import ixi

NUM_GAMES = 8
NUM_PLAYOUTS = 200
NUM_PREFILLED_EACH = 30


def main():
    assert len(sys.argv) == 2

    even_total = 0
    odd_total = 0
    draw_total = 0

    histories = [[] for _ in range(NUM_GAMES)]
    for game_index in range(NUM_GAMES):
        print("Game", game_index, "...")
        # Generate a game that's mostly full
        victory_state = 1
        while victory_state:
            to_sample = [1] * NUM_PREFILLED_EACH + [-1] * NUM_PREFILLED_EACH + [0] * (81 - 2 * NUM_PREFILLED_EACH)
            to_sample = np.array(to_sample, dtype=np.int8)
            if np.random.choice([0, 1]):
                to_sample[-1] = -1 
            np.random.shuffle(to_sample)
            ixi_ = to_sample.reshape((3, 3, 3, 3))
            victory_state = ixi.victory_state(ixi_)

        possible_prev_moves = np.transpose((ixi_ == -1).nonzero())
        prev_move = tuple(possible_prev_moves[np.random.randint(len(possible_prev_moves))])
        state_ = State(ixi_, prev_move)

        odd_mcts = Mcts(num_playouts=NUM_PLAYOUTS)
        # even_mcts = Mcts(num_playouts=10)
        even_mcts = odd_mcts

        for turn_index in itertools.count():
            histories[game_index].append(state_.copy())
            possible_moves = state_.list_valid_moves()
            values = []
            for possible_move in possible_moves:
                possible_state = state_.copy()
                possible_state.move(possible_move)
                value = (odd_mcts if turn_index % 2 else even_mcts).get_value(possible_state)
                value = -value  # Because we want to leave the next player in the worst-possible state
                values.append(value)
            state_.move(possible_moves[np.argmax(values)])
            victory_state = state_.victory_state()
            if victory_state:
                if victory_state == 2:
                    print("After", turn_index, "moves, draw.")
                    draw_total += 1
                else:
                    print("After", turn_index, "moves,", ("odd" if turn_index % 2 else "even"), "won.")
                    if turn_index % 2:
                        odd_total += 1
                    else:
                        even_total += 1
                break

    print(f"{even_total=}, {odd_total=}, {draw_total=}")

    with open(sys.argv[1], "wb") as f:
        pickle.dump(histories, f)


if __name__ == "__main__":
    main()
