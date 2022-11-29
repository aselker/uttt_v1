import os
import sys
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from nn_common import make_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Disable TensorFlow info messages, but not warnings or higher.
from tensorflow import keras

N_EPOCHS = 256
VAL_PORTION = 0.01
TEST_PORTION = 0.01
DROP_BEFORE = 6


def loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(from_logits=False, y_true=(y_true + 1.0) / 2.0, y_pred=(y_pred + 1.0) / 2.0)


def ingest_and_regurgitate(in_path, out_path):
    all_pairs = []
    filenames = Path(in_path).glob("**/*.pkl")
    for filename in filenames:
        with open(filename, "rb") as f:
            histories = pickle.load(f)

        for history in histories:

            # sEmAnTiC vErSiOnInG
            if len(history) == 2 and isinstance(history[0], tuple):
                history = history[1]

            if DROP_BEFORE is not None and len(history) > DROP_BEFORE:
                history = history[-DROP_BEFORE:]

            eventual_victory_state = history[-1].victory_state()
            if eventual_victory_state == 2:
                eventual_victory_state = 0
            eventual_victory_state = float(eventual_victory_state)
            for state_index, state_ in enumerate(history):
                # Parity: if the last state has victory state -1, then the last player to play won (of course).  So, the last state has value -1, since it's a losing state.  So, if len(history)==10, and state_index==9, it should not be inverted.
                value = eventual_victory_state if (len(history) - state_index) % 2 else -eventual_victory_state

                if False:  # No rotation/mirror augmentation
                    all_pairs.append((state_.ixi, value))
                else:
                    for rotation in [0, 1, 2, 3]:
                        all_pairs.append((np.rot90(state_.ixi, k=rotation), value))
                        all_pairs.append((np.rot90(state_.ixi.T, k=rotation), value))  # Mirrored

    with open(out_path, "wb") as f:
        pickle.dump(all_pairs, f)


def main():
    if sys.argv[1] == "preprocess":
        ingest_and_regurgitate(sys.argv[2], sys.argv[3])
        return

    with open(sys.argv[2], "rb") as f:
        all_pairs = pickle.load(f)

    np.random.shuffle(all_pairs)  # for plausible deniability
    all_inputs = np.array([pair[0] for pair in all_pairs])
    all_outputs = np.array([pair[1] for pair in all_pairs])
    # all_inputs = all_inputs.reshape(-1, 81)  # Flatten inputs.  For now.
    all_outputs = all_outputs[:, np.newaxis]  # For consistency, outputs are 1-lists.
    del all_pairs

    # Split into train and test
    train_inputs = all_inputs[: int(len(all_inputs) * (1 - TEST_PORTION))]
    test_inputs = all_inputs[int(len(all_inputs) * (1 - TEST_PORTION)) :]
    train_outputs = all_outputs[: int(len(all_outputs) * (1 - TEST_PORTION))]
    test_outputs = all_outputs[int(len(all_outputs) * (1 - TEST_PORTION)) :]
    print(len(train_inputs), "train pairs,", len(test_inputs), "test pairs")
    del (all_inputs, all_outputs)

    model = make_model()
    if len(sys.argv) == 4:
        model.load_weights(sys.argv[3])

    # 0.01 too high.  I think Keras defaults to 0.001.
    optimizer = keras.optimizers.Adam(learning_rate=0.000025)
    model.compile(optimizer=optimizer, loss=loss)

    tboard_callback = keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1, profile_batch="500,520")

    history = model.fit(
        train_inputs,
        train_outputs,
        epochs=N_EPOCHS,
        batch_size=2048,
        validation_split=VAL_PORTION,
        callbacks=[tboard_callback],
    )

    model.save_weights(sys.argv[1])

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.show()

    print("Test:")
    model.evaluate(
        test_inputs,
        test_outputs,
    )

    if True:  # Print some specific predictions
        print("Example test predictions:")
        preds = model.predict(test_inputs[:256])
        for pred, output in zip(preds[:32], test_outputs):
            print(pred, pred.round(), output)
        res = preds.round().astype(int) == test_outputs[: len(preds)]
        print(np.mean(res))


if __name__ == "__main__":
    main()
