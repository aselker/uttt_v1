import numpy as np
import ixi
import state
import mcts
import pickle


def main():
    with open("what.pkl", "rb") as f:
        history = pickle.load(f)

    mcts_ = mcts.Mcts(num_playouts=20)
    # for state_ in history:
    #     print(mcts_.get_value(state_))
    print(mcts_.get_value(history[-1]))
    print(mcts_.cache)
    mcts_.cache[hash(history[-1])]

if __name__ == "__main__":
    main()
