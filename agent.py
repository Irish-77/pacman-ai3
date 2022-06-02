from model import DQModelWithCNN
from options import Movement, Movements

import numpy as np

class Agent():
    def __init__(self, model:DQModelWithCNN, init_epsilon:float=0.99,
        epsilon_decay:float=0.997, epsilon_min:float=0.05) -> None:
        self.epsilon = init_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = model

    def predict_new_move_eps(self, observation) -> Movements:
        next_move = None

        random = np.random.rand()
        if random < self.epsilon:
            next_move = np.random.choice(list(Movements))
        else:
            next_move, _ = self.model.predict(observation) 

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        return next_move
    
    def predict_new_move(self, observation) -> Movements:
        next_move, _ = self.model.predict(observation)
        return next_move