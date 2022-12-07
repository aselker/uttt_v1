import sys
import pickle
import numpy as np
import itertools
from pathlib import Path
import warnings
import tensorflow as tf

from state import State
import ixi
from bots import MctsBot, SimpleNnBot, RandomBot, HumanBot, FasterMultiPlyNnBot

"""
Round-robin tournament.
"""

NUM_GAMES_PER_MATCHUP = 6
NUM_PREFILLED_EACH = 2
SOMETIMES_UNEQUAL = False
RUN_FOREVER = True


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


def summarize_histories(histories):
    # Calculate and display tournament results.
    # There's gotta be a better way to do this.  A dict isn't quite the right data structure.  Or maybe tournament_results should map from every pairing including duplicates.
    # tournament_results = {(bot1.name, bot2.name): [0, 0, 0] for bot1, bot2 in matchups}
    tournament_results = {}
    lengths = {}
    for history in histories:
        even_name, odd_name = history[0]
        if (even_name, odd_name) in tournament_results:
            relevant_results = tournament_results[(even_name, odd_name)]
            swap = False
        elif (odd_name, even_name) in tournament_results:
            relevant_results = tournament_results[(odd_name, even_name)]
            swap = True
        else:
            tournament_results[(even_name, odd_name)] = [0, 0, 0]
            relevant_results = tournament_results[(even_name, odd_name)]
            swap = False

        if history[1][-1].victory_state() == 2:
            relevant_results[2] += 1
        elif len(history[1]) % 2:
            if swap:
                relevant_results[0] += 1
            else:
                relevant_results[1] += 1
        else:
            if swap:
                relevant_results[1] += 1
            else:
                relevant_results[0] += 1

        lengths[len(history[1])] = lengths.get(len(history[1]), 0) + 1
    lengths = sorted([(key, lengths[key]) for key in lengths])  # I will not apologize
    print(tournament_results, "\nLengths:", lengths)


def main():
    if sys.argv[1] == "summarize":
        with open(sys.argv[2], "rb") as f:
            histories = pickle.load(f)
        summarize_histories(histories)
        return
    assert len(sys.argv) == 2

    for epoch in itertools.count():
        print("Starting epoch", epoch)
        bots = [
            FasterMultiPlyNnBot("training_data/trained_models/all.model", [99, 5]),
            FasterMultiPlyNnBot("training_data/trained_models/all.model", [99, 5], deterministic=True),
        ]

        # Make sure names are unique
        if not len(bots) == len({bot.name for bot in bots}):
            for i, bot in enumerate(bots):
                bot.name = str(i) + "_" + bot.name

        # All pairs, in alphabetical order
        matchups = list(itertools.combinations(bots, 2))

        # Play the round robin
        histories = []
        for bot1, bot2 in matchups:
            print("Starting matchup:", bot1.name, bot2.name)
            for game_index in range(NUM_GAMES_PER_MATCHUP):
                print("Starting game", game_index)
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

        # There's a race condition here!  Fortunately, nothing bad will happen.
        existing_pkls = set(Path(sys.argv[1]).glob("*.pkl"))
        for i in itertools.count():
            filename = Path(sys.argv[1]) / (str(i) + ".pkl")
            if filename not in existing_pkls:
                break

        with open(filename, "wb") as f:
            # pickle.dump([h[1] for h in histories], f)
            pickle.dump(histories, f)

        print("Written to", filename)

        summarize_histories(histories)

        if not RUN_FOREVER:
            break


if __name__ == "__main__":
    # tf.config.threading.set_intra_op_parallelism_threads(16)
    warnings.simplefilter("error", np.VisibleDeprecationWarning)
    main()
