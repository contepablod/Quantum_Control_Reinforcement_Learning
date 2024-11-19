import torch
import torch.nn as nn
from hyperparameters import config


# Dueling Double Deep Q-Network
class DDDQN(nn.Module):
    """
    Dueling Double Deep Q-Network (DDDQN) model.

    This model is used in reinforcement learning to estimate the Q-values of
    actions for a given state. The network uses a dueling architecture, where
    the value and advantage functions are separately estimated and combined
    to form the final Q-values.

    Attributes:
    -----------
    fc1 : nn.Linear
        The first fully connected layer that transforms the input state.
    fc2 : nn.Linear
        The second fully connected layer that further transforms the input.
    value_fc : nn.Linear
        The fully connected layer that estimates the state-value function.
    advantage_fc : nn.Linear
        The fully connected layer that estimates the advantage of each action.
    """

    def __init__(self, state_size, action_size):
        """
        Initializes the DDDQN model.

        Parameters:
        -----------
        state_size : int
            The size of the input state vector.
        action_size : int
            The number of possible actions.
        """
        super(DDDQN, self).__init__()
        self.fc1 = nn.Linear(
            state_size,
            config["hyperparameters"]["HIDDEN_FEATURES"]
            )
        self.fc2 = nn.Linear(
            config["hyperparameters"]["HIDDEN_FEATURES"],
            config["hyperparameters"]["HIDDEN_FEATURES"],
        )
        self.value_fc = nn.Linear(
            config["hyperparameters"]["HIDDEN_FEATURES"],
            1
        )
        self.advantage_fc = nn.Linear(
            config["hyperparameters"]["HIDDEN_FEATURES"], action_size
        )
        self.drop = nn.Dropout(p=config["hyperparameters"]["DROPOUT"])
        self._initialize_weights()

    def forward(self, x):
        """
        Defines the forward pass of the DDDQN model.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor representing the state.

        Returns:
        --------
        torch.Tensor
            The output tensor representing the Q-values for each action.
        """
        x = self.drop(torch.relu(self.fc1(x)))
        x = self.drop(torch.relu(self.fc2(x)))
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def _initialize_weights(self):
        """
        Initializes the weights of the model using Kaiming uniform
        initialization for the linear layers and zero initialization
        for the biases.
        """

        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

        self.apply(init_weights)


# Deep Q-Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(
            state_size,
            config["hyperparameters"]["HIDDEN_FEATURES"]
        )
        self.fc2 = nn.Linear(
            config["hyperparameters"]["HIDDEN_FEATURES"],
            config["hyperparameters"]["HIDDEN_FEATURES"],
        )
        self.fc3 = nn.Linear(
            config["hyperparameters"]["HIDDEN_FEATURES"],
            action_size
        )
        self.drop = nn.Dropout(p=config["hyperparameters"]["DROPOUT"])
        self._initialize_weights()

    def forward(self, state):
        x = self.drop(torch.relu(self.fc1(state)))
        x = self.drop(torch.relu(self.fc2(x)))
        return self.fc3(x)

    def _initialize_weights(self):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

        self.apply(init_weights)
