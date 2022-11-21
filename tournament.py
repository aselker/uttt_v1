import sys
import pickle
import numpy as np
import itertools
from pathlib import Path

from state import State
import ixi

from mcts import MctsBot, SimpleNnBot, RandomBot

"""
Round-robin tournament.
"""

NUM_GAMES_PER_MATCHUP = 16
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

    for epoch in itertools.count():
        bots = [MctsBot(30), MctsBot(35)]
        # bots = [MctsBot(30), SimpleNnBot("model.h5"), RandomBot()]
        # bots = [SimpleNnBot("model.h5"), SimpleNnBot("model.h5")]
        # bots[1].name += "_the_other"

        # Make sure names are unique
        assert len([bot.name for bot in bots]) == len({bot.name for bot in bots})

        # All pairs, in alphabetical order
        matchups = list(itertools.combinations(bots, 2))

        # Play the round robin
        histories = []
        for bot1, bot2 in matchups:
            for game_index in range(NUM_GAMES_PER_MATCHUP):
                if game_index % 2:
                    even_bot = bot1
                    odd_bot = bot2
                else:
                    even_bot = bot2
                    odd_bot = bot1

                state_ = generate_partially_full_state()
                history = ((even_bot.name, odd_bot.name), [state_.copy()])
                histories.append(history)

                for turn_index in itertools.count():
                    state_.move((odd_bot if turn_index % 2 else even_bot).get_move(state_))
                    history[1].append(state_.copy())
                    victory_state = state_.victory_state()
                    if victory_state:
                        break

        filename = Path(sys.argv[1]) / (str(epoch) + ".pkl")
        with open(filename, "wb") as f:
            # pickle.dump([h[1] for h in histories], f)
            pickle.dump(histories, f)

        # Calculate and display tournament results.
        # There's gotta be a better way to do this.  A dict isn't quite the right data structure.  Or maybe tournament_results should map from every pairing including duplicates.
        tournament_results = {(bot1.name, bot2.name): [0, 0, 0] for bot1, bot2 in matchups}
        for history in histories:
            even_name, odd_name = history[0]
            if (even_name, odd_name) in tournament_results:
                relevant_results = tournament_results[(even_name, odd_name)]
                swap = False
            else:
                relevant_results = tournament_results[(odd_name, even_name)]
                swap = True

            if history[1][-1].victory_state() == 2:
                relevant_results[2] += 1
            elif len(history) % 2:
                if swap:
                    relevant_results[0] += 1
                else:
                    relevant_results[1] += 1
            else:
                if swap:
                    relevant_results[1] += 1
                else:
                    relevant_results[0] += 1
        print(tournament_results)


if __name__ == "__main__":
    main()
