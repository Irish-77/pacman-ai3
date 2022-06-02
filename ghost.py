import numpy as np
import warnings
from options import Movements, Blocks

class Ghost():
    def __init__(self, start_pos, map) -> None:
        self.map = map
        self.current_pos = start_pos
        self.last_direction = None

    def get_possible_directions(self) -> list[Movements]:
        valid_moves = []

        for x in Movements:
            new_position = self.current_pos + x.direction
            if self.map[new_position[0], new_position[1]] != Blocks.WALL.id:
                valid_moves.append(x)

        return valid_moves

    def remove_last_opposite_direction(self, possible_directions) -> list:
        if self.last_direction is None:
            return possible_directions

        if self.last_direction == Movements.UP:
            possible_directions.remove(Movements.DOWN)
        elif self.last_direction == Movements.DOWN:
            possible_directions.remove(Movements.UP)
        elif self.last_direction == Movements.RIGHT:
            possible_directions.remove(Movements.LEFT)
        elif self.last_direction == Movements.LEFT:
            possible_directions.remove(Movements.RIGHT)
        else:
            warnings.warn(f'Error in MATRIX! Unknown direction: {self.last_direction}.')
            
        return possible_directions

    def move(self):
        possible_directions = self.get_possible_directions()

        # Overview of different situations:
        # x: path
        # ^: last direction of player
        # ---------------------------------------
        # |-----| |-----| |-----| |-----| |-----|
        # |  x  | |  x  | |  x  | |     | |     |
        # | x^x | |  ^x | |  ^  | |  ^x | |  ^  |
        # |  x  | |  x  | |  x  | |  x  | |  x  |
        # |-----| |-----| |-----| |-----| |-----|
        # Number of ways:
        #    4       3       2       2       1  

        if len(possible_directions) > 1:
            possible_directions = self.remove_last_opposite_direction(possible_directions)
        
        new_direction = np.random.choice(possible_directions)
        self.last_direction = new_direction
        self.current_pos = self.current_pos + new_direction.direction
