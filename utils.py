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
            return x.transpose(0, 2, 1)[..., ::-1, :]
        case 2:
            return x[..., ::-1, ::-1]
        case 3:
            return x.transpose(0, 2, 1)[..., :, ::-1]


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


# for rotation in [0, 1, 2, 3]:
#     rotated_ixi = utils.rotate_4d(state_.ixi, n=rotation)
#     rotated_prev_move = utils.rotate_2d(prev_move, n=rotation)
#     examples_in_which_to_save.append((rotated_ixi, rotated_prev_move, value))
#     examples_in_which_to_save.append((rotated_ixi.transpose(1, 0, 3, 2), rotated_prev_move.T, value))
#     n_total_pairs += 2

if __name__ == "__main__":
    # TODO: Time
    original = np.random.randint(10, size=(8, 3, 3), dtype=np.int8)
    a = rotate_2d(original)
    b = np.rot90(original, axes=(1, 2))
    assert np.all(a == b)
    original = np.random.randint(10, size=(8, 3, 3, 3, 3), dtype=np.int8)
    a = rotate_4d(original)
    b = np.rot90(np.rot90(original, axes=(1, 2)), axes=(3, 4))
    assert np.all(a == b)
