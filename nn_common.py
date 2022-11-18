from tensorflow import keras

l1 = 1e-7
l2 = 1e-6

def make_model():
    return keras.models.Sequential(
        # [keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.L1L2(l1=l1, l2=l2), input_shape=(81,))]
        # + [keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.L1L2(l1=l1, l2=l2)) for _ in range(7)]
        [keras.layers.Dense(256, activation="relu", input_shape=(81,))]
        + [keras.layers.Dense(256, activation="relu") for _ in range(7)]
        + [keras.layers.Dense(1, activation="tanh")]
    )
