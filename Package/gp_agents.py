import numpy as np
import torch
import torch.nn as nn
from hyperparameters import config
from networks import GRPO
from torch import GradScaler
from torch.distributions import Categorical, Normal, Independent
from torch.optim import AdamW


class BaseGPAgent:
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
        self.clip_epsilon = config["hyperparameters"]["PPO"]["CLIP_EPSILON"]
        self.entropy_coeff = config["hyperparameters"]["PPO"]["ENTROPY_COEFF"]

        # Model
        self.hidden_features = config["hyperparameters"]["general"]["HIDDEN_FEATURES"]
        self.dropout = config["hyperparameters"]["general"]["DROPOUT"]
        self.num_hidden_layers = config["hyperparameters"]["general"][
            "NUM_HIDDEN_LAYERS"
        ]


class GPAgent(BaseGPAgent):
    def __init__(self, env, agent_type, device, loss_type):
        super().__init__(
            env=env, agent_type=agent_type, device=device, loss_type=loss_type
        )

        # Optimizer
        self.lr = config["hyperparameters"]["optimizer"]["SCHEDULER_LEARNING_RATE"]
        self.weight_decay = config["hyperparameters"]["optimizer"]["WEIGHT_DECAY"]

        # Model
        self.grpo = GRPO(
            state_size=self.env.input_features,
            action_size=self.env.action_size,
            hidden_features=self.hidden_features,
            dropout=self.dropout,
            layer_norm=True,
            num_hidden_layers=self.num_hidden_layers,
            control_pulse_type=self.control_pulse_type,
            use_encoder=True,
        ).to(self.device)

        self.optimizer = AdamW(
            self.grpo.parameters(),
            lr=self.lr,
            amsgrad=True,
            weight_decay=self.weight_decay,
        )

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            if self.control_pulse_type == "Discrete":
                policy_probs = self.grpo(state)
                action_dist = Independent(Categorical(probs=policy_probs), 1)
                action = action_dist.sample().squeeze(0)
                log_prob = action_dist.log_prob(action).squeeze(0)
            elif self.control_pulse_type == "Continuous":
                mean, log_std = self.grpo(state)
                std = log_std.exp()
                action_dist = Independent(Normal(mean, std), 1)
                action = action_dist.sample().squeeze(0)
                log_prob = action_dist.log_prob(action).squeeze(0)

        return (
            action.cpu().numpy(),
            log_prob.cpu().numpy(),
        )

    def update(self, trajectory, epochs):

        states = trajectory["states"]
        actions = trajectory["actions"]
        old_log_probs = trajectory["log_probs"]
        dones = trajectory["dones"]
        advantages = trajectory["advantages"]

        # Convert to tensors
        states = torch.FloatTensor(states).detach().to(self.device)
        actions = torch.LongTensor(actions).detach().to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).detach().to(self.device)
        dones = torch.FloatTensor(dones).detach().to(self.device)
        advantages = torch.FloatTensor(advantages).detach().to(self.device)

        # Normalize advantages for stability
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = []
        pol_loss = []
        ent_loss = []

        for _ in range(epochs):
            # Compute loss
            loss, policy_loss, entropy_loss = self._compute_loss(
                old_log_probs,
                states,
                actions,
                advantages,
            )

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.grpo.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate loss for averaging
            total_loss.append(loss.item())
            pol_loss.append(policy_loss.item())
            ent_loss.append(entropy_loss.item())

        # Return average loss
        return (
            np.mean(total_loss),
            np.mean(pol_loss),
            np.mean(ent_loss),
        )

    def _compute_loss(
        self,
        old_log_probs,
        states,
        actions,
        advantages,
    ):
        # Forward pass to get new policy and value
        if self.control_pulse_type == "Discrete":
            policy_probs = self.grpo(states)
            action_dist = Independent(Categorical(probs=policy_probs), 1)
            log_probs = action_dist.log_prob(actions).squeeze(0)
            entropy = action_dist.entropy().mean()

        elif self.control_pulse_type == "Continuous":
            mean, log_std = self.grpo(states)
            std = log_std.exp()
            action_dist = Independent(Normal(mean, std), 1)
            log_probs = action_dist.log_prob(actions).squeeze(0)
            entropy = action_dist.entropy().mean()

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

        # Entropy loss
        entropy_loss = -self.entropy_coeff * entropy

        # Actor loss
        actor_loss = surrogate_loss + entropy_loss

        return actor_loss, surrogate_loss, entropy_loss
