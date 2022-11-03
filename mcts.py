import state


class MCTS:
    def __init__(num_playouts=20):
        # Playouts are mappings of {game state : pair of (average playout outcome where -1 is guaranteed loss, number of playouts that have been done)}
        self.playouts_cache = {}
        self.num_playouts = num_playouts

    def run_playouts(state):
        hash_ = ixi.hash(ixi_)
        if hash_ not in self.playouts_cache:
            self.playouts_cache[hash_] = (0.,0)

        while self.playouts_cache[hash_][1] < self.num_playouts:
            # Do a single playout.
            this_ixi = ixi_.copy()
            ixi.list_valid_moves(

