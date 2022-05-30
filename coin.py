from options import Blocks

class Coin():
    def __init__(self, position, state = Blocks.COIN.id) -> None:
        #position (y, x)
        self.position = position
        self.state = state

    def remove_coin_state(self):
        if self.state == Blocks.COIN.id:
            self.state = Blocks.EMPTY.id