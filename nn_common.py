from tensorflow import keras


def make_model():
    return keras.models.Sequential(
        [keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.L1L2(), input_shape=(81,))]
        + [keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.L1L2()) for _ in range(7)]
        + [keras.layers.Dense(1, activation="tanh")]
    )
