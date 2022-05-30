import numpy as np
from torch import rand
from options import Blocks, Movements
from model import DQModelWithCNN

class Pacman():
    def __init__(self,  init_position: np.array, map: np.array, init_epsilon:float = 0.99, epsilon_decay: float = 0.997) -> None:
        self.current_pos = init_position
        self.map = map
        self.number_of_actions = len(Movements)
        self.epsilon = init_epsilon
        self.epsilon_decay = epsilon_decay
        self.model = DQModelWithCNN()


    def predict_new_move(self, observation = None, reward = None):
        """
        observation: map (je nachdem wie viel er davon kennt)
        """
        next_move = None

        random = np.random.rand()
        if random < self.epsilon:
            next_move = np.random.choice(list(Movements))
        else:
            next_move, _ = self.model.predict(rand((1, 1, 21, 31))) 

        self.epsilon *= self.epsilon_decay

        self.move(next_move)

    def move(self, new_direction: Movements):
        new_position = self.current_pos + new_direction.direction
        if self.map[new_position[0], new_position[1]] != Blocks.WALL.id:    
            self.current_pos = new_position