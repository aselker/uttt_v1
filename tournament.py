import sys
import pickle
import numpy as np
import itertools

from state import State
import ixi

from mcts import MctsBot

"""
Round-robin tournament.
"""

NUM_GAMES_PER_MATCHUP = 8
NUM_PREFILLED_EACH = 30
SOMETIMES_UNEQUAL = False


def generate_partially_full_state():
    # Generate a game that's mostly full
    victory_state = 1
    while victory_state:
        to_sample = [1] * NUM_PREFILLED_EACH + [-1] * NUM_PREFILLED_EACH + [0] * (81 - 2 * NUM_PREFILLED_EACH)
        to_sample = np.array(to_sample, dtype=np.int8)
        if SOMETIMES_UNEQUAL and np.random.choice([0, 1]):
            to_sample[-1] = -1
        np.random.shuffle(to_sample)
        ixi_ = to_sample.reshape((3, 3, 3, 3))
        victory_state = ixi.victory_state(ixi_)

        possible_prev_moves = np.transpose((ixi_ == -1).nonzero())
        prev_move = tuple(possible_prev_moves[np.random.randint(len(possible_prev_moves))])
        state_ = State(ixi_, prev_move)
        return state_


def main():
    assert len(sys.argv) == 2

    bots = [MctsBot(200), MctsBot(100), MctsBot(30)]

    matchups = list(itertools.combinations(bots, 2))
    histories = []
    tournament_results = {}
    for bot1, bot2 in matchups:
        tournament_results[(bot1.name, bot2.name)] = (0, 0, 0)  # bot1, bot2, draw

        for game_index in range(NUM_GAMES_PER_MATCHUP):
            if game_index % 2:
                even_bot = bot1
                odd_bot = bot2
            else:
                even_bot = bot2
                odd_bot = bot1

            history = ((even_bot.name, odd_bot.name), [])
            histories.append(history)

            for turn_index in itertools.count():
                history[1].append(state_.copy())
                state_.move((odd_bot if turn_index % 2 else even_bot).get_move())

                victory_state = state_.victory_state()
                if victory_state:
                    history[1].append(state_.copy())
                    break

    # TODO: Print results

    with open(sys.argv[1], "wb") as f:
        pickle.dump(histories, f)


if __name__ == "__main__":
    main()
