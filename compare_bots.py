import numpy as np

from tournament import generate_partially_full_state
from bots import SimpleNnBot, FasterSimpleNnBot


def main():
    bots = [
        SimpleNnBot("training_data/trained_models/all.model"),
        FasterSimpleNnBot("training_data/trained_models/all.model"),
    ]

    state_ = generate_partially_full_state()
    while not state_.victory_state():
        # values = [bot.value_function(state_) for bot in bots]
        # if any(np.array(values) - abs(values[0]) > 0.0001):
        #     print("Different values:", {bot.name: value for bot, value in zip(bots, values)})
        moves = [bot.get_move(state_) for bot in bots]
        if any([move != moves[0] for move in moves]):
            print("Different moves:", {bot.name: move for bot, move in zip(bots, moves)})
        state_.move(moves[0])
        # return  # XXX


if __name__ == "__main__":
    main()
