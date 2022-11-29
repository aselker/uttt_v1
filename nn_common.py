import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Disable TensorFlow info messages, but not warnings or higher.
import tensorflow as tf
from tensorflow import keras
import numpy as np

l1 = 1e-7
l2 = 1e-6


def make_model():
    return keras.models.Sequential(
        # [keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.L1L2(l1=l1, l2=l2), input_shape=(81,))]
        # + [keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.L1L2(l1=l1, l2=l2)) for _ in range(7)]
        [
            keras.layers.Reshape((81,), input_shape=(3, 3, 3, 3)),
            keras.layers.Dense(256, activation="relu", input_shape=(81,)),
        ]
        + [keras.layers.Dense(256, activation="relu") for _ in range(5)]
        + [keras.layers.Dense(1, activation="tanh")]
    )


# class Symmetric(keras.units.Layer):
#     """
#     Input must be (batch size, input_dim, input_dim).  Output is (batch sizen, output_dim, output_dim):

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
