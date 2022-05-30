import numpy as np
from collections import namedtuple
from enum import Enum


Block = namedtuple('Block', ['value', 'color'])
class Blocks(Enum):
    # __order__ = 'EMPTY WALL PACMAN GHOST COIN GHOST_ON_COIN'

    @property
    def color(self):
        return self.value.color

    @property
    def id(self):
        return self.value.value

    EMPTY = Block(0, 'gray')
    WALL = Block(1, 'black')
    PACMAN = Block(2, 'orange')
    GHOST = Block(3, 'blue')
    COIN = Block(4, 'yellow')
    # GHOST_ON_COIN = Block(5, 'deepskyblue') # removed if not needed


Movement = namedtuple('Movement', ['value', 'direction'])
class Movements(Enum):
    # __order__ = 'UP DOWN LEFT RIGHT'

    @property
    def direction(self):
        return self.value.direction

    UP = Movement(1, np.array([-1, 0]))
    LEFT = Movement(2, np.array([0, -1]))
    DOWN = Movement(3, np.array([1, 0]))
    RIGHT = Movement(4, np.array([0, 1]))