import os
import sys
import pickle
import numpy as np
from nn_common import make_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Disable TensorFlow info messages, but not warnings or higher.
from tensorflow import keras

TEST_PORTION = 0.1


def main():
    histories = []
    for filename in sys.argv[2:]:
        with open(filename, "rb") as f:
            histories += pickle.load(f)

    all_pairs = []
    for history in histories:
        # sEmAnTiC vErSiOnInG
        if len(history) == 2 and isinstance(history[0], tuple):
            history = history[1]
        eventual_victory_state = history[-1].victory_state()
        assert eventual_victory_state, "Game ended without someone winning?"
        if eventual_victory_state == 2:
            eventual_victory_state = 0
        for state_index, state_ in enumerate(history):
            # Parity: if the last state has victory state -1, then the last player to play won (of course).  So, the last state has value -1, since it's a losing state.  So, if len(history)==10, and state_index==9, it should not be inverted.
            value = eventual_victory_state if (len(history) - state_index) % 2 else -eventual_victory_state
            all_pairs.append((state_.ixi, value))

    np.random.shuffle(all_pairs)  # for plausible deniability
    all_inputs = np.array([pair[0] for pair in all_pairs])
    all_outputs = np.array([pair[1] for pair in all_pairs])
    all_inputs = all_inputs.reshape(-1, 81)  # Flatten inputs.  For now.
    all_outputs = all_outputs[:, np.newaxis]  # For consistency, outputs are 1-lists.

    # Split into train and test
    train_inputs = all_inputs[: int(len(all_inputs) * (1 - TEST_PORTION))]
    test_inputs = all_inputs[int(len(all_inputs) * (1 - TEST_PORTION)) :]
    train_outputs = all_outputs[: int(len(all_outputs) * (1 - TEST_PORTION))]
    test_outputs = all_outputs[int(len(all_outputs) * (1 - TEST_PORTION)) :]
    print(len(train_inputs), "train pairs,", len(test_inputs), "test pairs")

    model = make_model()

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
    )

    model.evaluate(
        test_inputs,
        test_outputs,
    )

    model.fit(
        train_inputs,
        train_outputs,
        epochs=72,
        batch_size=256,
    )

    print("Test:")
    model.evaluate(
        test_inputs,
        test_outputs,
    )

    if True:  # Print some specific predictions
        print("Example test predictions:")
        preds = model.predict(test_inputs[:32])
        for pred, output in zip(preds, test_outputs):
            print(pred, pred.round(), output)
        res = preds.round().astype(int) == test_outputs[: len(preds)]
        print(np.mean(res))

    model.save_weights(sys.argv[1])


if __name__ == "__main__":
    main()
