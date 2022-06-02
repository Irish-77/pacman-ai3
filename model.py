import torch
import torch.nn as nn
import torch.nn.functional as F

from options import Movements

class DQModelWithCNN(nn.Module):

    def __init__(self, device:torch.device, height: int = 21, width: int = 31, number_of_actions: int = 4):
        self.device = device
        self.height = height
        self.width = width
        self.number_of_actions = number_of_actions

        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 6, 3)
        self.fc1 = nn.Linear(108, 200)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, self.number_of_actions)

    def forward(self, x):
        x = x.to(self.device)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_action(self, predictions) -> Movements:
        index = torch.argmax(predictions)
        return list(Movements)[index]

    def predict(self, x):
        self.eval()
        predictions = self.forward(x)
        return self.get_action(predictions), predictions

    def save(self, id:str):
        torch.save(self.state_dict(), f'models/{id}.model')

    def load(self, id:str):
        self.load_state_dict(torch.load(f'models/{id}.model'))