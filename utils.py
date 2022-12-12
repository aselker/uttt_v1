import numpy as np


def swap_players(x):
    """Edits the passed-in array!"""
    if isinstance(x, int):
        if x == -1:
            return 1
        elif x == 1:
            return -1
        return x

    was_1 = x == 1
    x[x == -1] = 1
    x[was_1] = -1
    return x


def rotate_2d(x, n=1):
    match n % 4:
        case 0:
            return x
        case 1:
            return x.T[..., ::-1, :]
        case 2:
            return x[..., ::-1, ::-1]
        case 3:
            return x.T[..., :, ::-1]


def rotate_4d(x, n=1):
    match n % 4:
        case 0:
            return x
        case 1:
            return x.transpose(0, -3, -4, -1, -2)[..., ::-1, :, ::-1, :]
        case 2:
            return x[..., ::-1, ::-1, ::-1, ::-1]
        case 3:
            return x.transpose(0, -3, -4, -1, -2)[..., :, ::-1, :, ::-1]


if __name__ == "__main__":
    # TODO: Time
    original = np.random.randint(10, size=(8, 3, 3), dtype=np.int8)
    a = rotate_2d(original)
    b = np.rot90(original, axes=(1, 2))
    original = np.random.randint(10, size=(8, 3, 3, 3, 3), dtype=np.int8)
    a = rotate_4d(original)
    b = np.rot90(np.rot90(original, axes=(1, 2)), axes=(3, 4))
    assert np.all(a == b)
