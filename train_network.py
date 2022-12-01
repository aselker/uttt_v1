import os
import sys
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from nn_common import make_model


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Disable TensorFlow info messages, but not warnings or higher.
import tensorflow as tf
from tensorflow import keras

# Preprocessing
DROP_BEFORE = None
LIMIT_EXAMPLE_COUNT = None

# Training
N_EPOCHS = 512
TEST_PORTION = 0.01
BATCH_SIZE = 8192
LEARN_RATE = 0.0003  # 0.01 too high.  I think Keras defaults to 0.001.  Karpathy constant == 0.0003


np.set_printoptions(precision=32)

# Allocate GPU memory dynamically.
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(from_logits=False, y_true=(y_true + 1.0) / 2.0, y_pred=(y_pred + 1.0) / 2.0)


def ingest_and_regurgitate(in_path, out_path):
    all_examples = []
    filenames = Path(in_path).glob("**/*.pkl")
    for filename in filenames:
        with open(filename, "rb") as f:
            histories = pickle.load(f)

        for history in histories:

            # sEmAnTiC vErSiOnInG
            if len(history) == 2 and isinstance(history[0], tuple):
                history = history[1]  # Drop bot names

            if DROP_BEFORE and len(history) > DROP_BEFORE:
                history = history[-DROP_BEFORE:]

            eventual_victory_state = history[-1].victory_state()
            if eventual_victory_state == 2:
                eventual_victory_state = 0
            eventual_victory_state = float(eventual_victory_state)
            for state_index, state_ in enumerate(history):
                # Parity: if the last state has victory state -1, then the last player to play won (of course).  So, the last state has value -1, since it's a losing state.  So, if len(history)==10, and state_index==9, it should not be inverted.
                value = eventual_victory_state if (len(history) - state_index) % 2 else -eventual_victory_state
                prev_move = np.zeros((3, 3))
                prev_move[state_.prev_move[2], state_.prev_move[3]] = 1
                for rotation in [0, 1, 2, 3]:
                    rotated_ixi = np.rot90(np.rot90(state_.ixi, axes=(2, 3), k=rotation), axes=(0, 1), k=rotation)
                    rotated_prev_move = np.rot90(prev_move, k=rotation)
                    all_examples.append((rotated_ixi, rotated_prev_move, value))
                    all_examples.append((rotated_ixi.transpose(1, 0, 3, 2), rotated_prev_move.T, value))  # Mirrored

    np.random.shuffle(all_examples)  # for plausible deniability

    if LIMIT_EXAMPLE_COUNT:
        all_examples = all_examples[:LIMIT_EXAMPLE_COUNT]

    ixis = np.array([example[0] for example in all_examples])
    prev_moves = np.array([example[1] for example in all_examples])
    results = np.array([example[2] for example in all_examples])

    with open(out_path, "wb") as f:
        np.savez(f, ixis=ixis, prev_moves=prev_moves, results=results)


class SequenceFromNumpyArrays(tf.keras.utils.Sequence):
    def __init__(self, batch_size, ixis, prev_moves, results):
        self.ixis = ixis
        self.prev_moves = prev_moves
        self.results = results
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.results) / self.batch_size))

    def __getitem__(self, index):
        index_slice = slice(index * self.batch_size, (index + 1) * self.batch_size)
        return [self.ixis[index_slice], self.prev_moves[index_slice]], self.results[index_slice]


def main():
    if sys.argv[1] == "preprocess":
        ingest_and_regurgitate(sys.argv[2], sys.argv[3])
        return

    with open(sys.argv[2], "rb") as f:
        loaded = np.load(f)
        all_ixis = loaded["ixis"]
        all_prev_moves = loaded["prev_moves"]
        all_results = loaded["results"]

    all_results = all_results[:, np.newaxis]  # For consistency, results are 1-lists.

    # Split into train and test
    train_count = int(len(all_ixis) * (1 - TEST_PORTION))
    train_ixis = all_ixis[:train_count]
    test_ixis = all_ixis[train_count:]
    train_prev_moves = all_prev_moves[:train_count]
    test_prev_moves = all_prev_moves[train_count:]
    train_results = all_results[:train_count]
    test_results = all_results[train_count:]
    print(len(train_ixis), "train pairs,", len(test_ixis), "test pairs")

    # # Explicitly move training data onto the CPU?
    # with tf.device("/CPU:0"):
    #     train_ixis = tf.convert_to_tensor(train_ixis)
    #     test_ixis = tf.convert_to_tensor(test_ixis)
    #     train_prev_moves = tf.convert_to_tensor(train_prev_moves)
    #     test_prev_moves = tf.convert_to_tensor(test_prev_moves)
    #     train_results = tf.convert_to_tensor(train_results)
    #     test_results = tf.convert_to_tensor(test_results)
    train_sequence = SequenceFromNumpyArrays(BATCH_SIZE, train_ixis, train_prev_moves, train_results)

    model = make_model()
    if len(sys.argv) == 4:
        model.load_weights(sys.argv[3])

    optimizer = keras.optimizers.Adam(learning_rate=LEARN_RATE)
    model.compile(optimizer=optimizer, loss=loss)

    # tboard_callback = keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1, profile_batch="500,520")

    history = model.fit(
        train_sequence,
        epochs=N_EPOCHS,
        validation_data=([test_ixis, test_prev_moves], test_results),
        # callbacks=[tboard_callback],
    )

    model.save_weights(sys.argv[1])

    if False:  # Plot history
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.show()

    print("Test:")
    model.evaluate(
        [test_ixis, test_prev_moves],
        test_results,
    )

    if True:  # Print some specific predictions
        print("Example test predictions:")
        preds = model.predict([test_ixis[:512], test_prev_moves[:512]])
        for pred, result in zip(preds[:32], test_results):
            print(pred, pred.round(), result)
        res = preds.round().astype(int) == test_results[: len(preds)]
        print(np.mean(res))


if __name__ == "__main__":
    main()
