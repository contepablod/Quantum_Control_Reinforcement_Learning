import torch.nn as nn
import torch


# Dueling Double Deep Q-Network
class DDDQN(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_features,
        dropout,
        layer_norm=False,
        num_hidden_layers=1,
    ):
        super(DDDQN, self).__init__()

        # Shared encoder: num_hidden_layers - 1
        encoder_layers = []
        input_size = state_size
        for _ in range(num_hidden_layers):
            encoder_layers.append(nn.Linear(input_size, hidden_features))
            if layer_norm:
                encoder_layers.append(nn.LayerNorm(hidden_features))
            encoder_layers.append(nn.ReLU(inplace=False))
            encoder_layers.append(nn.Dropout(p=dropout))
            input_size = hidden_features
        self.encoder = nn.Sequential(*encoder_layers)

        # Advantage stream: num_hidden_layers
        advantage_layers = []
        input_size = hidden_features
        for _ in range(num_hidden_layers):
            advantage_layers.append(nn.Linear(input_size, hidden_features))
            if layer_norm:
                advantage_layers.append(nn.LayerNorm(hidden_features))
            advantage_layers.append(nn.ReLU(inplace=False))
            advantage_layers.append(nn.Dropout(p=dropout))
            input_size = hidden_features
        advantage_layers.append(nn.Linear(hidden_features, action_size))
        self.advantage_stream = nn.Sequential(*advantage_layers)

        # Value stream: num_hidden_layers
        value_layers = []
        input_size = hidden_features
        for _ in range(num_hidden_layers):
            value_layers.append(nn.Linear(input_size, hidden_features))
            if layer_norm:
                value_layers.append(nn.LayerNorm(hidden_features))
            value_layers.append(nn.ReLU(inplace=False))
            value_layers.append(nn.Dropout(p=dropout))
            input_size = hidden_features
        value_layers.append(nn.Linear(hidden_features, 1))
        self.value_stream = nn.Sequential(*value_layers)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        encoded = self.encoder(x)
        value = self.value_stream(encoded)
        advantage = self.advantage_stream(encoded)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def _initialize_weights(self, init_type="kaiming_uniform"):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                if init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                elif init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(layer.weight)
                else:
                    raise ValueError(f"Unsupported init_type: {init_type}")

                # Set biases to zero
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.apply(init_weights)


class Actor_TD3(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_features,
        dropout,
        layer_norm=False,
        num_hidden_layers=1,
        use_encoder=True,
        control_pulse_type=None,
        max_action=2
    ):
        super(Actor_TD3, self).__init__()
        self.action_space = control_pulse_type
        self.use_encoder = use_encoder
        self.max_action = max_action

        if use_encoder:
            self.actor_encoder = self._build_encoder(
                state_size, hidden_features, dropout, layer_norm, num_hidden_layers
            )
        else:
            # No encoder; direct input to streams
            self.actor_input_size = state_size

        # Actor stream (policy network)
        self.actor_stream = nn.Sequential(
                nn.Linear(
                    hidden_features if use_encoder else state_size, hidden_features
                ),
                nn.LayerNorm(hidden_features) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=False),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_features, action_size),

            )
        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        if self.use_encoder:
            actor_encoded = self.actor_encoder(x)
        else:
            actor_encoded = x
        x = self.actor_stream(actor_encoded)
        return self.max_action * torch.tanh(x)

    def _build_encoder(
        self, state_size, hidden_features, dropout, layer_norm, num_hidden_layers
    ):
        layers = []
        input_size = state_size
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_size, hidden_features))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_features))
            layers.append(nn.ReLU(inplace=False))
            layers.append(nn.Dropout(p=dropout))
            input_size = hidden_features
        return nn.Sequential(*layers)

    def _initialize_weights(self, init_type="kaiming_uniform"):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                if init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(
                        layer.weight, mode="fan_in", nonlinearity="relu"
                    )
                elif init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_in", nonlinearity="relu"
                    )
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(layer.weight)
                else:
                    raise ValueError(f"Unsupported init_type: {init_type}")

                # Set biases to zero
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.apply(init_weights)


class Critic_TD3(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_features,
        dropout,
        layer_norm=False,
        num_hidden_layers=1,
        use_encoder=True,
    ):
        super(Critic_TD3, self).__init__()
        self.use_encoder = use_encoder

        if use_encoder:
            self.critic_encoder = self._build_encoder(
                state_size + action_size, hidden_features, dropout, layer_norm, num_hidden_layers
            )
        else:
            self.critic_input_size = state_size + action_size

        # Critic stream (value network)
        self.critic_stream = nn.Sequential(
            nn.Linear(hidden_features if use_encoder else state_size + action_size, hidden_features),
            nn.LayerNorm(hidden_features) if layer_norm else nn.Identity(),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, 1),
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        if self.use_encoder:
            critic_encoded = self.critic_encoder(x)

        else:
            critic_encoded = x

        value = self.critic_stream(critic_encoded)
        return value

    def _build_encoder(
        self, state_size, hidden_features, dropout, layer_norm, num_hidden_layers
    ):
        """Helper function to build an encoder."""
        layers = []
        input_size = state_size
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_size, hidden_features))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_features))
            layers.append(nn.ReLU(inplace=False))
            layers.append(nn.Dropout(p=dropout))
            input_size = hidden_features
        return nn.Sequential(*layers)

    def _initialize_weights(self, init_type="kaiming_uniform"):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                if init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(
                        layer.weight, mode="fan_in", nonlinearity="relu"
                    )
                elif init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_in", nonlinearity="relu"
                    )
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(layer.weight)
                else:
                    raise ValueError(f"Unsupported init_type: {init_type}")

                # Set biases to zero
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.apply(init_weights)


# Actor-Critic
class PPO(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_features,
        dropout,
        layer_norm=False,
        num_hidden_layers=3,
        use_encoder=True,
        control_pulse_type=None,
    ):
        super(PPO, self).__init__()
        self.action_space = control_pulse_type
        self.use_encoder = use_encoder

        if use_encoder:
                # Shared encoder
                self.shared_encoder = self._build_encoder(
                    state_size,
                    hidden_features,
                    dropout,
                    layer_norm,
                    num_hidden_layers
                )
        else:
            # No encoder; direct input to streams
            self.actor_input_size = state_size
            self.critic_input_size = state_size

        # Actor stream (policy network)
        if self.action_space == "Discrete":
            self.actor_stream = nn.Sequential(
                nn.Linear(
                    hidden_features if use_encoder else state_size, hidden_features
                ),
                nn.LayerNorm(hidden_features) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=False),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_features, action_size),
                nn.Softmax(dim=-1),
            )
        elif self.action_space == "Continuous":
            self.actor_mean = nn.Sequential(
                nn.Linear(
                    hidden_features if use_encoder else state_size, hidden_features
                ),
                nn.LayerNorm(hidden_features) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=False),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_features, action_size),
            )
            self.actor_log_std = nn.Sequential(
                nn.Linear(
                    hidden_features if use_encoder else state_size, hidden_features
                ),
                nn.LayerNorm(hidden_features) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=False),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_features, action_size),
            )

        # Critic stream (value network)
        self.critic_stream = nn.Sequential(
            nn.Linear(hidden_features if use_encoder else state_size, hidden_features),
            nn.LayerNorm(hidden_features) if layer_norm else nn.Identity(),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, 1),
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        if self.use_encoder:
            encoded = self.shared_encoder(x)
            actor_encoded = critic_encoded = encoded
        else:
            actor_encoded = critic_encoded = x

        if self.action_space == "Discrete":
            action_probs = self.actor_stream(actor_encoded)
            value = self.critic_stream(critic_encoded)
            return action_probs, value
        elif self.action_space == "Continuous":
            action_mean = self.actor_mean(actor_encoded)
            action_log_std = self.actor_log_std(actor_encoded)
            value = self.critic_stream(critic_encoded)
            return action_mean, action_log_std, value
        
    def _build_encoder(
        self,
        state_size,
        hidden_features,
        dropout,
        layer_norm,
        num_hidden_layers
    ):
        """Helper function to build an encoder."""
        layers = []
        input_size = state_size
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_size, hidden_features))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_features))
            layers.append(nn.ReLU(inplace=False))
            layers.append(nn.Dropout(p=dropout))
            input_size = hidden_features
        return nn.Sequential(*layers)

    def _initialize_weights(self, init_type="kaiming_uniform"):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                if init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                elif init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(layer.weight)
                else:
                    raise ValueError(f"Unsupported init_type: {init_type}")

                # Set biases to zero
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.apply(init_weights)


class GRPO(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_features,
        dropout,
        layer_norm=False,
        num_hidden_layers=3,
        use_encoder=True,
        control_pulse_type=None,
    ):
        super(GRPO, self).__init__()
        self.action_space = control_pulse_type
        self.use_encoder = use_encoder

        if use_encoder:
            self.shared_encoder = self._build_encoder(
                    state_size, hidden_features, dropout, layer_norm, num_hidden_layers
                )
        else:
            # No encoder; direct input to streams
            self.actor_input_size = state_size

        # Actor stream (policy network)
        if self.action_space == "Discrete":
            self.actor_stream = nn.Sequential(
                nn.Linear(
                    hidden_features if use_encoder else state_size, hidden_features
                ),
                nn.LayerNorm(hidden_features) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=False),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_features, action_size),
                nn.Softmax(dim=-1),
            )
        elif self.action_space == "Continuous":
            self.actor_mean = nn.Sequential(
                nn.Linear(
                    hidden_features if use_encoder else state_size, hidden_features
                ),
                nn.LayerNorm(hidden_features) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=False),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_features, action_size),
            )
            self.actor_log_std = nn.Sequential(
                nn.Linear(
                    hidden_features if use_encoder else state_size, hidden_features
                ),
                nn.LayerNorm(hidden_features) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=False),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_features, action_size),
            )

        # Initialize weights
        self._initialize_weights()

    def _build_encoder(
        self, state_size, hidden_features, dropout, layer_norm, num_hidden_layers
    ):
        """Helper function to build an encoder."""
        layers = []
        input_size = state_size
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_size, hidden_features))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_features))
            layers.append(nn.ReLU(inplace=False))
            layers.append(nn.Dropout(p=dropout))
            input_size = hidden_features
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.use_encoder:
            encoded = self.shared_encoder(x)
            actor_encoded = encoded
        else:
            actor_encoded = x
        if self.action_space == "Discrete":
            action_probs = self.actor_stream(actor_encoded)
            return action_probs
        elif self.action_space == "Continuous":
            action_mean = self.actor_mean(actor_encoded)
            action_log_std = self.actor_log_std(actor_encoded)
            return action_mean, action_log_std

    def _initialize_weights(self, init_type="kaiming_uniform"):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                if init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                elif init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(layer.weight)
                else:
                    raise ValueError(f"Unsupported init_type: {init_type}")

                # Set biases to zero
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.apply(init_weights)


# Actor Network
class Actor(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_features,
        dropout,
        layer_norm=False,
        num_hidden_layers=1,
        use_encoder=True,
        control_pulse_type=None,
    ):
        super(Actor, self).__init__()
        self.action_space = control_pulse_type
        self.use_encoder = use_encoder

        if use_encoder:
            self.actor_encoder = self._build_encoder(
                state_size, hidden_features, dropout, layer_norm, num_hidden_layers
            )
        else:
            # No encoder; direct input to streams
            self.actor_input_size = state_size

        # Actor stream (policy network)
        if self.action_space == "Discrete":
            self.actor_stream = nn.Sequential(
                nn.Linear(
                    hidden_features if use_encoder else state_size, hidden_features
                ),
                nn.LayerNorm(hidden_features) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=False),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_features, action_size),
                nn.Softmax(dim=-1),
            )
        elif self.action_space == "Continuous":
            self.actor_mean = nn.Sequential(
                nn.Linear(
                    hidden_features if use_encoder else state_size, hidden_features
                ),
                nn.LayerNorm(hidden_features) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=False),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_features, action_size),
            )
            self.actor_log_std = nn.Sequential(
                nn.Linear(
                    hidden_features if use_encoder else state_size, hidden_features
                ),
                nn.LayerNorm(hidden_features) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=False),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_features, action_size),
            )
            # Learnable log standard deviation
            # self.actor_log_std = nn.Parameter(torch.zeros(action_size))

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        if self.use_encoder:
            actor_encoded = self.actor_encoder(x)
        else:
            actor_encoded = x
        if self.action_space == "Discrete":
            action_probs = self.actor_stream(actor_encoded)
            return action_probs
        elif self.action_space == "Continuous":
            action_mean = self.actor_mean(actor_encoded)
            action_log_std = self.actor_log_std(actor_encoded)
            return action_mean, action_log_std

    def _build_encoder(
        self, state_size, hidden_features, dropout, layer_norm, num_hidden_layers
    ):
        layers = []
        input_size = state_size
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_size, hidden_features))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_features))
            layers.append(nn.ReLU(inplace=False))
            layers.append(nn.Dropout(p=dropout))
            input_size = hidden_features
        return nn.Sequential(*layers)

    def _initialize_weights(self, init_type="kaiming_uniform"):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                if init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(
                        layer.weight, mode="fan_in", nonlinearity="relu"
                    )
                elif init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_in", nonlinearity="relu"
                    )
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(layer.weight)
                else:
                    raise ValueError(f"Unsupported init_type: {init_type}")

                # Set biases to zero
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.apply(init_weights)


# Critic Network
class Critic(nn.Module):

    def __init__(
        self,
        state_size,
        hidden_features,
        dropout,
        layer_norm=False,
        num_hidden_layers=1,
        use_encoder=True,
    ):
        super(Critic, self).__init__()
        self.use_encoder = use_encoder

        if use_encoder:
            self.critic_encoder = self._build_encoder(
                state_size, hidden_features, dropout, layer_norm, num_hidden_layers
            )
        else:
            self.critic_input_size = state_size

        # Critic stream (value network)
        self.critic_stream = nn.Sequential(
            nn.Linear(hidden_features if use_encoder else state_size, hidden_features),
            nn.LayerNorm(hidden_features) if layer_norm else nn.Identity(),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, 1),
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        if self.use_encoder:
            critic_encoded = self.critic_encoder(x)
        else:
            critic_encoded = x
        value = self.critic_stream(critic_encoded)
        return value

    def _build_encoder(
        self, state_size, hidden_features, dropout, layer_norm, num_hidden_layers
    ):
        """Helper function to build an encoder."""
        layers = []
        input_size = state_size
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_size, hidden_features))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_features))
            layers.append(nn.ReLU(inplace=False))
            layers.append(nn.Dropout(p=dropout))
            input_size = hidden_features
        return nn.Sequential(*layers)

    def _initialize_weights(self, init_type="kaiming_uniform"):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                if init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(
                        layer.weight, mode="fan_in", nonlinearity="relu"
                    )
                elif init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_in", nonlinearity="relu"
                    )
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(layer.weight)
                else:
                    raise ValueError(f"Unsupported init_type: {init_type}")

                # Set biases to zero
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.apply(init_weights)


# Deep Q-Network
class DQN(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_features,
        dropout,
        layer_norm=False,
        num_hidden_layers=1,
    ):
        super(DQN, self).__init__()
        self.layer_norm = layer_norm

        # Build layers dynamically
        layers = []
        input_size = state_size
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_size, hidden_features))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_features))
            layers.append(nn.ReLU(inplace=False))
            layers.append(nn.Dropout(p=dropout))
            input_size = hidden_features

        # Fully connected layers
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_features, action_size)

        # Initialize weights
        self._initialize_weights()

    def forward(self, state):
        x = self.hidden_layers(state)
        return self.output_layer(x)

    def _initialize_weights(self, init_type="kaiming_uniform"):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                if init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(
                        layer.weight, mode="fan_in", nonlinearity="relu"
                    )
                elif init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_in", nonlinearity="relu"
                    )
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(layer.weight)
                else:
                    raise ValueError(f"Unsupported init_type: {init_type}")

                # Set biases to zero
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.apply(init_weights)
