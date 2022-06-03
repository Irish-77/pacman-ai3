import torch
import torch.nn as nn
import torch.nn.functional as F

from options import Movements

class DQModelWithCNN(nn.Module):
    """Deep Q Neuronal Network with CNN"""

    def __init__(self, device:torch.device, height:int=21, width:int=31,
        number_of_actions:int=4):
        """Constructor

        Args:
            device (torch.device): Device on which the computation will be done.
            height (int, optional): Height of the input tensor. Defaults to 21.
            width (int, optional): Width of the input tensor. Defaults to 31.
            number_of_actions (int, optional): Number of possible actions to
                take. Defaults to 4.
        """

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

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Forward Pass

        Passes the inputs through the model.

        Args:
            x (torch.Tensor): Input for the model.

        Returns:
            torch.Tensor: Output predictions of the model. These are q-values as
                in Deep Q-Networks.
        """
        x = x.to(self.device)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_action(self, predictions:torch.Tensor) -> Movements:
        """Get Action based on Q-Values

        The maximum expected future reward is choosen which is represented by
        the predicted q-values of the model.

        Args:
            predictions (torch.Tensor): Predictions of the model itself. The
                size should be equal to the number of actions.

        Returns:
            Movements: Direction to take based on the selected action.
        """
        index = torch.argmax(predictions)
        return list(Movements)[index]

    def predict(self, x:torch.Tensor) -> tuple[Movements, torch.Tensor]:
        """Predict next Action to take

        Args:
            x (torch.Tensor): Input for the model.

        Returns:
            tuple[Movements, torch.Tensor]:
                [0]: Direction to take based on the selected action.
                [1]: Output predictions of the model. These are q-values as in
                    Deep Q-Networks.
        """
        self.eval()
        predictions = self.forward(x)
        return self.get_action(predictions), predictions

    def save(self, id:str) -> None:
        """Save the current State of the Model

        Creates a new '.model' file inside the 'models' folder.

        Args:
            id (str): An unique identifier used to identify the model without
                its file ending.
        """
        torch.save(self.state_dict(), f'models/{id}.model')

    def load(self, id:str) -> None:
        """Load a specific State for the Model

        Reads the model parameters from a '.model' file inside the 'models'
        folder.

        Args:
            id (str): An unique identifier used to identify the model without
                its file ending.
        """
        self.load_state_dict(torch.load(f'models/{id}.model'))