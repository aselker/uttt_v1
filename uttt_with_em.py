# Written with @proceduralcode

import itertools


def empty_board():
    return [[0 for x in range(9)] for y in range(9)]


symbols = ["  ", "x ", "o "]


def print_board(board):
    print()
    for y, row in enumerate(board):
        print("  ", end="")
        for x, cell in enumerate(row):
            print(symbols[cell], end="")
            if x in (2, 5):
                print("| ", end="")
        print()
        if y in (2, 5):
            print("  ------+-------+------")


def cxc_victory_state(state):
    # sorry
    for i in range(3):
        if state[0 + 3 * i] == state[1 + 3 * i] == state[2 + 3 * i]:
            if state[0 + 3 * i] != " ":
                return state[0 + 3 * i]
        if state[0 + i] == state[3 + i] == state[6 + i]:
            if state[0 + i] != " ":
                return state[0 + i]

    if state[0] == state[4] == state[8]:
        if state[0] != " ":
            return state[0]

    if state[2] == state[4] == state[6]:
        if state[2] != " ":
            return state[2]

    if " " in state:
        return None
    return "c"


cxc_goodnesses = {}


states = itertools.product("xo ", repeat=9)
buckets = [[] for _ in range(10)]
for state in states:
    buckets[len([None for i in state if i == " "])].append("".join(state))

for bucket in buckets:
    print(len(bucket))
    for state in bucket:
        victory_state = cxc_victory_state(state)
        if victory_state is not None:
            value = {"x": 1, "o": -1, "c": 0}[victory_state]
        else:
            # The hard part
            empty_spots = [i for i in range(9) if state[i] == " "]
            total_value = 0  # Don't ask what the units of this is
            for empty_spot in empty_spots:
                this_state = state[:empty_spot] + "x" + state[empty_spot + 1 :]
                total_value += cxc_goodnesses[this_state]
                this_state = state[:empty_spot] + "o" + state[empty_spot + 1 :]
                total_value += cxc_goodnesses[this_state]
            value = total_value / (len(empty_spots) * 2)
        cxc_goodnesses[state] = value


for key, val in cxc_goodnesses.items():
    print(f"{key}: {val}")


def square_health():
    return float


board = empty_board()
print_board(board)
