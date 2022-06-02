import numpy as np
from options import Blocks, Movements

class Pacman():
    def __init__(self,  init_position: np.array, map: np.array):
        self.current_pos = init_position
        self.map = map

    def move(self, new_direction: Movements):
        new_position = self.current_pos + new_direction.direction
        if self.map[new_position[0], new_position[1]] != Blocks.WALL.id:    
            self.current_pos = new_position