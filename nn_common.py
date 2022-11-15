from tensorflow import keras

def make_model():
    return keras.models.Sequential(
    [
        keras.layers.Dense(81, activation="relu", input_shape=(81,)),
        keras.layers.Dense(81, activation="relu"),
        keras.layers.Dense(81, activation="relu"),
        keras.layers.Dense(81, activation="relu"),
        keras.layers.Dense(81, activation="relu"),
        keras.layers.Dense(81, activation="relu"),
        keras.layers.Dense(81, activation="relu"),
        keras.layers.Dense(1, activation="tanh"),
    ]
)
