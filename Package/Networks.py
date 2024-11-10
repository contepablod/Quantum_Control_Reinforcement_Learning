import torch.nn as nn


# Dueling Double Deep Q-Network
class DDDQN(nn.Module):
    def __init__(self, state_size, action_size, dropout=0.1):
        super(DDDQN, self).__init__()
        self.fc1 = nn.Linear(env.input_features, env.hidden_features)
        self.fc2 = nn.Linear(env.hidden_features, env.hidden_features)
        self.value_fc = nn.Linear(env.hidden_features, 1)
        self.advantage_fc = nn.Linear(env.hidden_features, env.action_size)
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def _initialize_weights(self):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)
        self.apply(init_weights)
