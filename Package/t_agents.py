import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from hyperparameters import config
from memory import ReplayMemory
from networks import Actor_TD3, Critic_TD3
from torch import GradScaler
from torch.optim import AdamW


class BaseTAgent:
    def __init__(self, env, agent_type, device, loss_type):

        # Environment
        self.env = env

        # Agent
        self.agent_type = agent_type

        # Hyperparameters
        self.device = device
        self.control_pulse_type = self.env.control_pulse_type

        # Loss
        self.loss_type = loss_type
        if self.loss_type == "HUBER":
            self.loss = nn.HuberLoss().to(self.device)
        else:
            self.loss = nn.MSELoss().to(self.device)

        # Gradient scaler
        self.actor_scaler = GradScaler(device=self.device.type)
        self.grad_clip = False

        # Additional hyperparameters for policy-based methods
        self.tau = config["hyperparameters"]["TD3"]["TAU"]
        self.noise = config["hyperparameters"]["TD3"]["NOISE"]
        self.policy_noise = config["hyperparameters"]["TD3"]["POLICY_NOISE"]
        self.noise_clip = config["hyperparameters"]["TD3"]["NOISE_CLIP"]
        self.gamma = config["hyperparameters"]["general"]["GAMMA"]

        # Model
        self.hidden_features = config["hyperparameters"]["general"][
            "HIDDEN_FEATURES"
            ]
        self.dropout = config["hyperparameters"]["general"]["DROPOUT"]
        self.num_hidden_layers = config["hyperparameters"]["general"][
            "NUM_HIDDEN_LAYERS"
        ]


class TD3Agent(BaseTAgent):
    def __init__(self, env, agent_type, device, loss_type):
        super().__init__(
            env=env, agent_type=agent_type, device=device, loss_type=loss_type
        )

        # Memory and model
        self.memory = ReplayMemory(config["hyperparameters"]["DDDQN"][
            "MEMORY_SIZE"]
            )
        self.batch_size = config["hyperparameters"]["general"]["BATCH_SIZE"]

        # Optimizer
        self.lr = config["hyperparameters"]["optimizer"][
            "SCHEDULER_LEARNING_RATE"
            ]
        self.weight_decay = config["hyperparameters"]["optimizer"][
            "WEIGHT_DECAY"
            ]

        # Model
        self.actor = Actor_TD3(
            state_size=self.env.input_features,
            action_size=self.env.action_size,
            hidden_features=self.hidden_features,
            dropout=self.dropout,
            layer_norm=True,
            num_hidden_layers=self.num_hidden_layers,
            control_pulse_type=self.control_pulse_type,
            use_encoder=True,
        ).to(self.device)

        self.critic_1 = Critic_TD3(
            state_size=self.env.input_features,
            action_size=self.env.action_size,
            hidden_features=self.hidden_features,
            dropout=self.dropout,
            layer_norm=True,
            num_hidden_layers=self.num_hidden_layers,
            use_encoder=True,
        ).to(self.device)

        self.critic_2 = Critic_TD3(
            state_size=self.env.input_features,
            action_size=self.env.action_size,
            hidden_features=self.hidden_features,
            dropout=self.dropout,
            layer_norm=True,
            num_hidden_layers=self.num_hidden_layers,
            use_encoder=True,
        ).to(self.device)

        self.actor_optimizer = AdamW(
            self.actor.parameters(),
            lr=self.lr,
            amsgrad=True,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = AdamW(
            list(self.critic_1.parameters())+list(self.critic_2.parameters()),
            lr=self.lr * 2,
            amsgrad=True,
            weight_decay=self.weight_decay,
        )

        # Counter for exponential decay
        self.current_update = 0

        self.actor_target = deepcopy(self.actor).to(self.device)
        self._update_actor_model()

        self.critic_1_target = deepcopy(self.critic_1).to(self.device)
        self._update_critic_1_model()

        self.critic_2_target = deepcopy(self.critic_2).to(self.device)
        self._update_critic_2_model()

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()
            action += np.random.normal(0, self.noise, size=action.shape)
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        # Ensure replay memory has enough samples
        while len(self.memory) < self.batch_size:
            state = self.env.reset()
            for _ in range(self.env.max_steps):
                # Generate a random action to explore
                action = self.act(state)
                (
                    done,
                    next_state,
                    step_reward,
                ) = self.env.step(action)
                # Add experience to memory
                self.remember(state, action, step_reward, next_state, done)
                # Update current state
                state = next_state
                # Stop when the episode ends
                if done:
                    break

        # Sample from replay memory
        (states, actions, rewards, next_states, dones) = self.memory.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Target Policy Smoothing
        noise = (torch.randn_like(actions) * self.policy_noise).clamp(
            -self.noise_clip, self.noise_clip
        )
        next_actions = (self.actor_target(next_states) + noise)

        # Compute target Q-values
        target_q1 = self.critic_1_target(next_states, next_actions)
        target_q2 = self.critic_2_target(next_states, next_actions)
        rewards = rewards.view(-1, 1)  # Ensures shape [64, 1]
        dones = dones.view(-1, 1)  # Ensures shape [64, 1]      
        target_q = rewards + self.gamma * (1 - dones) * torch.min(
            target_q1,
            target_q2
        )

        # Update Critics
        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        loss = self.loss(q1, target_q.detach()) + self.loss(
            q2, target_q.detach()
        )

        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        if np.random.randint(0, 2) == 0:  # Delayed updates
            actor_loss = -self.critic_1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update Target Networks
            self._update_target_networks()

        return loss.item()

    def _update_actor_model(self):
        self.actor_target.load_state_dict(self.actor.state_dict())

    def _update_critic_1_model(self):
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())

    def _update_critic_2_model(self):
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

    def _update_target_networks(self):
        # Update actor target network
        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        # Update first critic target network
        for param, target_param in zip(
            self.critic_1.parameters(), self.critic_1_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        # Update second critic target network
        for param, target_param in zip(
            self.critic_2.parameters(), self.critic_2_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
