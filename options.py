import numpy as np
from collections import namedtuple
from enum import Enum


Block = namedtuple('Block', ['value', 'color'])
class Blocks(Enum):

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


Movement = namedtuple('Movement', ['value', 'direction'])
class Movements(Enum):

    @property
    def direction(self):
        return self.value.direction

    @property
    def id(self):
        return self.value.value

    UP = Movement(0, np.array([-1, 0]))
    LEFT = Movement(1, np.array([0, -1]))
    DOWN = Movement(2, np.array([1, 0]))
    RIGHT = Movement(3, np.array([0, 1]))