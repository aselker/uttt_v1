import numpy as np
import ixi
import state
import mcts
import random


def main():
    np.random.seed(0)
    ixi_ = np.zeros((3, 3, 3, 3), dtype=np.uint8)
    # ixi_ = np.random.choice([-1, 0, 1], size=(3, 3, 3, 3))
    print(ixi.pretty_print(ixi_))
    state_ = state.State(ixi_=ixi_)
    mcts_ = mcts.Mcts()
    mcts_.run_playouts(state_)
    print(mcts_.cache[hash(state_)])

    ixi_ = np.zeros((3, 3, 3, 3), dtype=np.uint8)
    ixi_[1,1,:,:] = 2
    print(ixi.pretty_print(ixi_))
    state_ = state.State(ixi_=ixi_)
    mcts_ = mcts.Mcts()
    mcts_.run_playouts(state_)
    print(mcts_.cache[hash(state_)])


if __name__ == "__main__":
    main()
