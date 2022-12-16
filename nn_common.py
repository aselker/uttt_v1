import os
import numpy as np
import state
import gc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Disable TensorFlow info messages, but not warnings or higher.
import tensorflow as tf
from tensorflow import keras


# Allocate GPU memory dynamically.
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def make_model():
    regularizer = None
    # regularizer = kernel_regularizer=keras.regularizers.L1L2(l1=0.01, l2=0.01)

    arm_thickness = 32

    ixi_input = keras.Input(shape=(3, 3, 3, 3))
    flattened_once = keras.layers.Reshape((3, 3, 9))(ixi_input)

    cxc_layers = [keras.layers.Dense(arm_thickness, activation="relu", kernel_regularizer=regularizer) for _ in range(4)]
    cxc_outputs = []
    for i in range(3):
        for j in range(3):
            cxc_intermediate = flattened_once[:, i, j]
            for layer in cxc_layers:
                cxc_intermediate = layer(cxc_intermediate)
            cxc_outputs.append(cxc_intermediate)

    prev_move_input = keras.Input(shape=(3, 3))
    prev_move_flattened = keras.layers.Reshape((9,))(prev_move_input)
    cxc_outputs_and_prev_move = keras.layers.Concatenate()(cxc_outputs + [prev_move_flattened])
    ixi_intermediate = keras.layers.Reshape(((arm_thickness * 9) + 9,))(cxc_outputs_and_prev_move)
    for _ in range(8):
        ixi_intermediate = keras.layers.Dense(256, activation="relu", kernel_regularizer=regularizer)(ixi_intermediate)
    for _ in range(16):
        ixi_intermediate = keras.layers.Dense(192, activation="relu", kernel_regularizer=regularizer)(ixi_intermediate)
    output = keras.layers.Dense(1, activation="tanh", kernel_regularizer=regularizer)(ixi_intermediate)

    return keras.Model([ixi_input, prev_move_input], output)


def make_older_model():
    # Compatible with  2022_12_14_larger.model
    regularizer = None
    # regularizer = kernel_regularizer=keras.regularizers.L1L2(l1=0.01, l2=0.01)

    arm_thickness = 32

    ixi_input = keras.Input(shape=(3, 3, 3, 3))
    flattened_once = keras.layers.Reshape((3, 3, 9))(ixi_input)

    cxc_layers = [keras.layers.Dense(arm_thickness, activation="relu", kernel_regularizer=regularizer) for _ in range(4)]
    cxc_outputs = []
    for i in range(3):
        for j in range(3):
            cxc_intermediate = flattened_once[:, i, j]
            for layer in cxc_layers:
                cxc_intermediate = layer(cxc_intermediate)
            cxc_outputs.append(cxc_intermediate)

    prev_move_input = keras.Input(shape=(3, 3))
    prev_move_flattened = keras.layers.Reshape((9,))(prev_move_input)
    cxc_outputs_and_prev_move = keras.layers.Concatenate()(cxc_outputs + [prev_move_flattened])
    ixi_intermediate = keras.layers.Reshape(((arm_thickness * 9) + 9,))(cxc_outputs_and_prev_move)
    for _ in range(16):
        ixi_intermediate = keras.layers.Dense(192, activation="relu", kernel_regularizer=regularizer)(ixi_intermediate)
    output = keras.layers.Dense(1, activation="tanh", kernel_regularizer=regularizer)(ixi_intermediate)

    return keras.Model([ixi_input, prev_move_input], output)


def call_model_on_states(model, states):
    if isinstance(states, state.State):
        was_single = True
        states = [states]
    else:
        was_single = False

    prev_moves = np.zeros((len(states), 3, 3))
    for i, state_ in enumerate(states):
        prev_moves[i, state_.prev_move[2], state_.prev_move[3]] = 1
    prediction = model.predict([np.array([state_.ixi for state_ in states]), prev_moves], verbose=False)[:, 0]

    if np.random.rand() < 0.01:
        keras.backend.clear_session()
        gc.collect()

    if was_single:
        return prediction[0]
    else:
        return prediction
