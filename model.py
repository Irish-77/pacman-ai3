
import torch.nn as nn
import torch.nn.functional as F

from options import Movements
from torch import flatten, argmax, rand
from torchsummary import summary



class DQModelWithCNN(nn.Module):

    # TODO: create custom input channel

    def __init__(self, height: int = 21, width: int = 31, number_of_actions: int = 4):
        self.height = height
        self.width = width
        self.number_of_actions = number_of_actions

        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3) #in_channels, out_channels, kernel_size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 6, 3)
        self.fc1 = nn.Linear(108, 200)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, self.number_of_actions)
        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_action(self, predictions) -> Movements:
        index = argmax(predictions)
        return list(Movements)[index]

    def predict(self, x):
        self.eval()
        predictions = self.forward(x)
        # predictions [0.99, 1.9, 4.9, 3]
        return self.get_action(predictions), predictions



net = DQModelWithCNN()
print(net.predict(rand((1, 1, 21, 31))))
summary(net, (1, 21, 31)) #channels, height, width
