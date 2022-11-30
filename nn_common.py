import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Disable TensorFlow info messages, but not warnings or higher.
import tensorflow as tf
from tensorflow import keras
import numpy as np


def make_model():
    regularizer = None
    # regularizer = kernel_regularizer=keras.regularizers.L1L2(l1=1e-7, l2=1e-6)

    if False:
        # Simple fully-connected
        return keras.models.Sequential(
            [
                keras.layers.Reshape((81,), input_shape=(3, 3, 3, 3)),
                keras.layers.Dense(256, activation="relu", input_shape=(81,)),
            ]
            + [keras.layers.Dense(256, activation="relu") for _ in range(5)]
            + [keras.layers.Dense(1, activation="tanh")]
        )
    else:
        arm_thickness = 32

        ixi_input = keras.Input(shape=(3, 3, 3, 3))
        flattened_once = keras.layers.Reshape((3, 3, 9))(ixi_input)

        cxc_layers = [keras.layers.Dense(arm_thickness, activation="relu") for _ in range(4)]
        cxc_outputs = []
        for i in range(3):
            for j in range(3):
                cxc_intermediate = flattened_once[:, i, j]
                for layer in cxc_layers:
                    cxc_intermediate = layer(cxc_intermediate)
                cxc_outputs.append(cxc_intermediate)

        prev_move_input = keras.Input(shape=(3, 3))
        cxc_outputs_and_prev_move = keras.layers.Concatenate()(cxc_outputs + [prev_move_input])
        ixi_intermediate = keras.layers.Reshape((arm_thickness * 9,))(cxc_outputs_concatenated)
        for _ in range(4):
            ixi_intermediate = keras.layers.Dense(256, activation="relu")(ixi_intermediate)
        output = keras.layers.Dense(1, activation="tanh")(ixi_intermediate)

        return keras.Model([ixi_input, prev_move_input], output)


# class Symmetric(keras.units.Layer):
#     """
#     Input must be (batch size, input_dim, input_dim).  Output is (batch size, output_dim, output_dim):

#     Stored weights and biases are (almost) twice as big as they need to be.  I think this is better than "folding" the
#     triangle into a rectangle.

#     """

#     def __init__(self, input_dim=9, output_dim=12):
#         super(Symmetric, self).__init__()

#         # w_init = tf.random_normal_initializer()
#         # w_triangle = tf.Variable(
#         #         initial_value=w_init(

#         half_input_dim = int(np.ceil(input_dim / 2))
#         half_output_dim = int(np.ceil(output_dim / 2))

#         w_init = tf.random_normal_initializer()
#         w_small = tf.Variable(
#             initial_value=w_init(
#                 shape=(half_input_dim, half_input_dim, half_output_dim, half_output_dim)
#             ),
#             trainable=True,
#         )
#         b_init = tf.zeros_initializer()
#         b_small = tf.Variable(initial_value=b_init(shape=(half_output_dim, half_output_dim)), trainable=True)

#         input_diagonal_mask = np.triu(np.ones((half_input_dim, half_input_dim)))
#         output_diagonal_mask = np.triu(np.ones((half_output_dim, half_output_dim)))


#         w = tf.concat(
#                 input_diagonal_mask


#     def call(self, input_):
#         tf.einsum("iab,abcd->icd", a, b)  # XXX
