import sys
import pickle
import numpy as np


def main():
    histories = []
    for filename in sys.argv[1:]:
        with open(sys.argv[1], "rb") as f:
            histories += pickle.load(f)

    breakpoint()

if __name__ == "__main__":
    main()
