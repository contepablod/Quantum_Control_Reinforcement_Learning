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
        advantage = self.advantage_stream(encoded)
        # advantage = advantage.view(-1, 2, 5)
        advantage_mean = advantage.mean(dim=-1, keepdim=True)
        value = self.value_stream(encoded)
        # value = value.view(-1, 1, 1).expand(-1, 2, 5)
        q_values = value + (advantage - advantage_mean)
        return q_values

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


# PPO-Actor/Critic
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
        self.action_size = action_size
        self.action_space = control_pulse_type
        self.use_encoder = use_encoder

        if use_encoder:
            # Shared encoder
            self.shared_encoder = self._build_encoder(
                state_size, hidden_features, dropout, layer_norm, num_hidden_layers
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
                nn.Linear(hidden_features, 5 * action_size),
                # nn.Softmax(dim=-1),
            )
        elif self.action_space == "Continuous":
            # self.actor_stream = nn.Sequential(
            #     nn.Linear(
            #         hidden_features if use_encoder else state_size, hidden_features
            #     ),
            #     nn.LayerNorm(hidden_features) if layer_norm else nn.Identity(),
            #     nn.ReLU(inplace=False),
            #     nn.Dropout(p=dropout),
            #     nn.Linear(hidden_features, 2 * action_size),
            #     # nn.Softmax(dim=-1),
            # )
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
            action_logits = self.actor_stream(actor_encoded)
            actions_logits = action_logits.view(-1, 5, self.action_size)
            action_probs = torch.softmax(actions_logits, dim=-1)
            value = self.critic_stream(critic_encoded)
            return value, action_probs
        elif self.action_space == "Continuous":
            action_mean = self.actor_mean(actor_encoded)
            #action_mean = torch.clamp(action_mean, min=-2.5, max=2.5)
            action_log_std = self.actor_log_std(actor_encoded)
            action_log_std = torch.clamp(action_log_std, min=-20, max=2)
            value = self.critic_stream(critic_encoded)
            return value, action_mean, action_log_std

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
        self.action_size = action_size
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
                nn.Linear(hidden_features, 5 * action_size),
                # nn.Softmax(dim=-1),
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
            action_logits = self.actor_stream(actor_encoded)
            actions_logits = action_logits.view(-1, 5, self.action_size)
            action_probs = torch.softmax(actions_logits, dim=-1)
            return action_probs
        elif self.action_space == "Continuous":
            action_mean = self.actor_mean(actor_encoded)
            # action_mean = torch.clamp(action_mean, min=-2.5, max=2.5)
            action_log_std = self.actor_log_std(actor_encoded)
            action_log_std = torch.clamp(action_log_std, min=-20, max=2)
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
        max_action=2,
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
            nn.Linear(hidden_features if use_encoder else state_size, hidden_features),
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
                state_size + action_size,
                hidden_features,
                dropout,
                layer_norm,
                num_hidden_layers,
            )
        else:
            self.critic_input_size = state_size + action_size

        # Critic stream (value network)
        self.critic_stream = nn.Sequential(
            nn.Linear(
                hidden_features if use_encoder else state_size + action_size,
                hidden_features,
            ),
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


# Actor Network
class Actor_PPO(nn.Module):
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
        super(Actor_PPO, self).__init__()
        self.action_space = control_pulse_type
        self.use_encoder = use_encoder

        if use_encoder:
            # Shared encoder
            self.shared_encoder = self._build_encoder(
                state_size, hidden_features, dropout, layer_norm, num_hidden_layers
            )
        else:
            # No encoder; direct input to streams
            self.actor_input_size = state_size

        # Actor stream (policy network)
        self.actor_stream = nn.Sequential(
            nn.Linear(hidden_features if use_encoder else state_size, hidden_features),
            nn.LayerNorm(hidden_features) if layer_norm else nn.Identity(),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, 2 * action_size),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_features if use_encoder else state_size, hidden_features),
            nn.LayerNorm(hidden_features) if layer_norm else nn.Identity(),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, action_size),
        )
        self.actor_log_std = nn.Sequential(
            nn.Linear(hidden_features if use_encoder else state_size, hidden_features),
            nn.LayerNorm(hidden_features) if layer_norm else nn.Identity(),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, action_size),
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        if self.use_encoder:
            encoded = self.shared_encoder(x)
            actor_encoded = encoded
        else:
            actor_encoded = x

        action_mean = self.actor_mean(actor_encoded)
        action_log_std = self.actor_log_std(actor_encoded)
        action_log_std = torch.clamp(action_log_std, min=-20, max=-10)
        # output = self.actor_stream(actor_encoded)
        # action_mean, action_log_std = output.chunk(2, dim=-1)
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


# Critic Network
class Critic_PPO(nn.Module):
    def __init__(
        self,
        state_size,
        hidden_features,
        dropout,
        layer_norm=False,
        num_hidden_layers=3,
        use_encoder=True,
        control_pulse_type=None,
    ):
        super(Critic_PPO, self).__init__()
        self.action_space = control_pulse_type
        self.use_encoder = use_encoder

        if use_encoder:
            # Shared encoder
            self.shared_encoder = self._build_encoder(
                state_size, hidden_features, dropout, layer_norm, num_hidden_layers
            )
        else:
            # No encoder; direct input to streams
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
            encoded = self.shared_encoder(x)
            critic_encoded = encoded
        else:
            critic_encoded = x

        value = self.critic_stream(critic_encoded)
        return value

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
