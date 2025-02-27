import math
import numpy as np
import torch
import torch.nn as nn
from hyperparameters import config
from networks import PPO
from torch import GradScaler
from torch.distributions import Categorical, Normal, Independent
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ExponentialLR,
    CosineAnnealingWarmRestarts
)


class BasePAgent:
    def __init__(
            self,
            env,
            agent_type,
            device,
            loss_type
            ):

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
        self.clip_epsilon = config["hyperparameters"]["PPO"]["CLIP_EPSILON"]
        self.value_coeff = config["hyperparameters"]["PPO"]["VALUE_COEFF"]
        self.entropy_coeff = config["hyperparameters"]["DDDQN"]["MAX_EPSILON"]
        self.epsilon_min = config["hyperparameters"]["DDDQN"]["MIN_EPSILON"]
        self.epsilon_max = config["hyperparameters"]["DDDQN"]["MAX_EPSILON"]
        self.epsilon_decay_rate = config["hyperparameters"]["DDDQN"][
            "EPSILON_DECAY_RATE"
        ]

        # Model
        self.hidden_features = config["hyperparameters"]["general"][
            "HIDDEN_FEATURES"
            ]
        self.dropout = config["hyperparameters"]["general"][
            "DROPOUT"
            ]
        self.num_hidden_layers = config["hyperparameters"]["general"][
            "NUM_HIDDEN_LAYERS"
            ]


class PPOAgent(BasePAgent):
    def __init__(
            self,
            env,
            agent_type,
            device,
            scheduler_type,
            loss_type
    ):
        super().__init__(
            env=env,
            agent_type=agent_type,
            device=device,
            loss_type=loss_type
        )

        # Optimizer
        self.scheduler_type = scheduler_type
        self.lr = config["hyperparameters"]["optimizer"][
            "SCHEDULER_LEARNING_RATE"
            ]
        self.weight_decay = config["hyperparameters"]["optimizer"][
            "WEIGHT_DECAY"
            ]
        self.exp_decay = config["hyperparameters"]["optimizer"]["EXP_LR_DECAY"]

        # Model

        self.ppo = PPO(
            state_size=self.env.input_features,
            action_size=self.env.action_size,
            hidden_features=self.hidden_features,
            dropout=self.dropout,
            layer_norm=True,
            num_hidden_layers=self.num_hidden_layers,
            use_encoder=True,
            control_pulse_type=self.control_pulse_type
        ).to(self.device)

        self.optimizer = AdamW(
            self.ppo.parameters(),
            lr=self.lr,
            amsgrad=True,
            weight_decay=self.weight_decay,
        )

        # Learning rate scheduler
        if self.scheduler_type == "EXP":
            self.lr_scheduler = ExponentialLR(
                self.optimizer, gamma=self.exp_decay
            )
        elif self.scheduler_type == "COS":
            self.lr_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config["hyperparameters"]["COS_WARMUP_STEPS"],
                T_mult=config["hyperparameters"]["COS_WARMUP_FACTOR"],
                eta_min=config["hyperparameters"]["SCHEDULER_LR_MIN"],
            )

        # Counter for exponential decay
        self.current_update = 0

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            if self.control_pulse_type == "Discrete":
                policy, value = self.ppo(state)
                action_dist = Categorical(policy)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            elif self.control_pulse_type == "Continuous":
                mean, log_std, value = self.ppo(state)
                log_std = torch.clamp(log_std, min=-0.5, max=0.5)
                std = log_std.exp()
                action_dist = Independent(Normal(mean, std), 1)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)  #.sum(dim=-1)
        return (
            action.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy()
        )

    def update(self, trajectory, epochs):
        # Extract trajectory elements
        states = trajectory['states']
        actions = trajectory['actions']
        returns = trajectory['returns']
        old_log_probs = trajectory['log_probs']
        dones = trajectory['dones']
        advantages = trajectory['advantages']

        # Convert to tensors
        states = torch.FloatTensor(states).detach().to(self.device)
        actions = torch.LongTensor(actions).detach().to(self.device)
        returns = torch.FloatTensor(returns).detach().to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).detach().to(self.device)
        dones = torch.FloatTensor(dones).detach().to(self.device)
        advantages = torch.FloatTensor(advantages).detach().to(self.device)

        # Normalize advantages for stability
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = []
        surr_loss = []
        critic_loss = []
        ent_loss = []

        for _ in range(epochs):
            # Compute loss
            loss, surrogate_loss, value_loss, entropy_loss = self._compute_loss(
                old_log_probs,
                states,
                actions,
                returns,
                advantages,
            )

            # Backpropagate
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.ppo.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate loss for averaging
            total_loss.append(loss.item())
            surr_loss.append(surrogate_loss.item())
            critic_loss.append(value_loss.item())
            ent_loss.append(entropy_loss.item())

        # Update entropy coefficient
        self.entropy_coeff = self.epsilon_min + (
            (self.epsilon_max - self.epsilon_min)
            * math.exp(-self.epsilon_decay_rate * self.current_update)
        )

        self.current_update += 1

        # Return average loss over epochs
        return np.mean(total_loss), np.mean(surr_loss), np.mean(critic_loss), np.mean(ent_loss)

    def _compute_loss(
        self,
        old_log_probs,
        states,
        actions,
        returns,
        advantages
    ):
        # Forward pass to get new policy and value
        if self.control_pulse_type == "Discrete":
            policy, values = self.ppo(states)
            log_probs = torch.log(policy.gather(1, actions.unsqueeze(-1).long()) + 1e-10)
            action_dist = Categorical(policy)
            entropy = action_dist.entropy().mean()

        elif self.control_pulse_type == "Continuous":
            mean, log_std, values = self.ppo(states)
            log_std = torch.clamp(log_std, min=-0.5, max=0.5)
            std = log_std.exp()
            action_dist = Independent(Normal(mean, std), 1)
            log_probs = action_dist.log_prob(actions)   #.sum(dim=-1)
            entropy = action_dist.entropy().mean()  #.sum(dim=-1).mean()

        # Compute PPO ratios
        ratios = torch.exp(log_probs - old_log_probs)

        # Clipped surrogate objective
        clipped_ratios = torch.clamp(
            ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon
        )

        # Surrogate loss
        surrogate_loss = -torch.min(
            ratios * advantages, clipped_ratios * advantages
        ).mean()

        # Critic loss
        critic_loss = self.value_coeff * self.loss(values.view_as(returns), returns)

        # Entropy loss
        entropy_loss = -self.entropy_coeff * entropy

        # Combine losses
        total_loss = surrogate_loss + critic_loss + entropy_loss

        return total_loss, surrogate_loss, critic_loss, entropy_loss
