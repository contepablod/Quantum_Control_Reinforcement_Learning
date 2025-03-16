import numpy as np
import torch
import torch.nn as nn
from hyperparameters import config
from networks import PPO, Actor_PPO, Critic_PPO
from torch import GradScaler
from torch.distributions import Categorical, Normal, Independent
from torch.optim import AdamW


class BasePAgent:
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
        self.value_coeff = config["hyperparameters"]["PPO"]["VALUE_COEFF"]
        self.entropy_coeff = config["hyperparameters"]["PPO"]["ENTROPY_COEFF"]

        # Model
        self.hidden_features = config["hyperparameters"]["general"]["HIDDEN_FEATURES"]
        self.dropout = config["hyperparameters"]["general"]["DROPOUT"]
        self.num_hidden_layers = config["hyperparameters"]["general"][
            "NUM_HIDDEN_LAYERS"
        ]


class PPOAgent(BasePAgent):
    def __init__(self, env, agent_type, device, loss_type):
        super().__init__(
            env=env, agent_type=agent_type, device=device, loss_type=loss_type
        )

        # Optimizer
        self.lr = config["hyperparameters"]["optimizer"]["SCHEDULER_LEARNING_RATE"]
        self.weight_decay = config["hyperparameters"]["optimizer"]["WEIGHT_DECAY"]

        self.ppo = PPO(
            state_size=self.env.input_features,
            action_size=self.env.action_size,
            hidden_features=self.hidden_features,
            dropout=self.dropout,
            layer_norm=True,
            num_hidden_layers=self.num_hidden_layers,
            use_encoder=True,
            control_pulse_type=self.control_pulse_type,
        ).to(self.device)

        self.actor = Actor_PPO(
            state_size=self.env.input_features,
            action_size=self.env.action_size,
            hidden_features=self.hidden_features,
            dropout=self.dropout,
            layer_norm=True,
            num_hidden_layers=self.num_hidden_layers,
            use_encoder=True,
            control_pulse_type=self.control_pulse_type,
        ).to(self.device)

        self.critic = Critic_PPO(
            state_size=self.env.input_features,
            hidden_features=self.hidden_features,
            dropout=self.dropout,
            layer_norm=True,
            num_hidden_layers=self.num_hidden_layers,
            use_encoder=True,
        ).to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.ppo.parameters(),
            lr=self.lr,
            amsgrad=True,
            weight_decay=self.weight_decay,
        )

        self.actor_optimizer = AdamW(
            self.actor.parameters(),
            lr=self.lr,
            amsgrad=True,
            weight_decay=self.weight_decay,
        )

        self.critic_optimizer = AdamW(
            self.critic.parameters(),
            lr=2 * self.lr,
            amsgrad=True,
        )

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            if self.control_pulse_type == "Discrete":
                value, policy_probs = self.ppo(state)
                action_dist = Independent(Categorical(probs=policy_probs), 1)
                action = action_dist.sample().squeeze(0)
                log_prob = action_dist.log_prob(action).squeeze(0)

            elif self.control_pulse_type == "Continuous":
                value, mean, log_std = self.ppo(state)
                std = log_std.exp()
                action_dist = Independent(Normal(mean, std), 1)
                action = action_dist.sample().squeeze(0)
                log_prob = action_dist.log_prob(action).squeeze(0)
                # mean, log_std = self.actor(state)
                # value = self.critic(state)
        
        return (action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy())

    def update(self, trajectory, epochs):
        # Extract trajectory elements
        states = trajectory["states"]
        actions = trajectory["actions"]
        old_log_probs = trajectory["log_probs"]
        dones = trajectory["dones"]
        returns = trajectory["returns"]
        advantages = trajectory["advantages"]

        # Convert to tensors
        states = torch.FloatTensor(states).detach().to(self.device)
        actions = torch.LongTensor(actions).detach().to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).detach().to(self.device)
        dones = torch.FloatTensor(dones).detach().to(self.device)
        returns = torch.FloatTensor(returns).detach().to(self.device)
        advantages = torch.FloatTensor(advantages).detach().to(self.device)

        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = []
        pol_loss = []
        critic_loss = []
        ent_loss = []

        for _ in range(epochs):
            # Compute loss
            (loss, policy_loss, value_loss, entropy_loss) = self._compute_loss(
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

            # IF ACTOR and CRITIC ARE SPLITTED
            # self.actor_optimizer.zero_grad(set_to_none=True)
            # (surrogate_loss + entropy_loss).backward()
            # if self.grad_clip:
            #     nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            # self.actor_optimizer.step()

            # self.critic_optimizer.zero_grad(set_to_none=True)
            # value_loss.backward()
            # if self.grad_clip:
            #     nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            # self.critic_optimizer.step()

            # Accumulate loss for averaging
            total_loss.append(loss.item())
            pol_loss.append(policy_loss.item())
            critic_loss.append(value_loss.item())
            ent_loss.append(entropy_loss.item())

        # Return average loss over epochs
        return (
            np.mean(total_loss),
            np.mean(pol_loss),
            np.mean(critic_loss),
            np.mean(ent_loss),
        )

    def _compute_loss(self, old_log_probs, states, actions, returns, advantages):
        # Forward pass to get new policy and value
        if self.control_pulse_type == "Discrete":
            values, policy_probs = self.ppo(states)
            values = values.squeeze(-1)
            action_dist = Independent(Categorical(probs=policy_probs), 1)
            log_probs = action_dist.log_prob(actions).squeeze(0)
            entropy = action_dist.entropy().mean()

        elif self.control_pulse_type == "Continuous":
            values, mean, log_std = self.ppo(states)
            values = values.squeeze(-1)
            # mean, log_std = self.actor(states)
            std = log_std.exp()
            action_dist = Independent(Normal(mean, std), 1)
            log_probs = action_dist.log_prob(actions).squeeze(0)
            entropy = action_dist.entropy().mean()

            # values = self.critic(states)

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
        critic_loss = self.value_coeff * self.loss(values, returns)

        # Entropy loss
        entropy_loss = -self.entropy_coeff * entropy

        # Combine losses
        total_loss = surrogate_loss + critic_loss + entropy_loss

        return total_loss, surrogate_loss, critic_loss, entropy_loss
