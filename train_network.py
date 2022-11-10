import sys
import pickle
import numpy as np
from tensorflow import keras

TEST_PORTION = 0.1


def main():
    histories = []
    for filename in sys.argv[1:]:
        with open(sys.argv[1], "rb") as f:
            histories += pickle.load(f)

    # XXX
    print(len([h for h in histories if len(h) % 2]))
    print(len([h for h in histories if len(h) % 2 == 0]))
    breakpoint()

    all_pairs = []
    for history in histories:
        eventual_victory_state = history[-1].victory_state()
        assert eventual_victory_state, "Game ended without someone winning?"
        if eventual_victory_state == 3:
            eventual_victory_state = 0
        for state_index, state_ in enumerate(history):
            # Parity: if the last state has victory state -1, then the last player to play won (of course).  So, the last state has value -1, since it's a losing state.  So, if len(history)==10, and state_index==9, it should not be inverted.
            value = eventual_victory_state if (len(history) - state_index) % 2 else -eventual_victory_state
            all_pairs.append(state_.ixi, eventual_victory_state)

    np.random.shuffle(all_pairs)  # for plausible deniability
    train_pairs = all_pairs[: int(len(all_pairs) * (1 - TEST_PORTION))]
    test_pairs = all_pairs[int(len(all_pairs) * (1 - TEST_PORTION)) :]

    model = keras.models.Sequential(
        [
            keras.layers.Dense(81, activation="relu", use_bias=True, input_shape=(3, 3, 3, 3)),
            keras.layers.Dense(81, activation="relu", use_bias=True),
            keras.layers.Dense(81, activation="relu", use_bias=True),
            keras.layers.Dense(1, activation=None),
        ]
    )


if __name__ == "__main__":
    main()
