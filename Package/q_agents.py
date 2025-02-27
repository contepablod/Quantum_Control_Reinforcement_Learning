import math
import numpy as np
import random
import torch
import torch.nn as nn
from copy import deepcopy
from memory import ReplayMemory
from hyperparameters import config
from networks import DDDQN
from torch import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ExponentialLR,
    CosineAnnealingWarmRestarts
)


class BaseQAgent:

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
        self.scaler = GradScaler(device=self.device.type)

        # Epsilon decay
        self.epsilon = config["hyperparameters"]["DDDQN"]["MAX_EPSILON"]
        self.epsilon_min = config["hyperparameters"]["DDDQN"]["MIN_EPSILON"]
        self.epsilon_max = config["hyperparameters"]["DDDQN"]["MAX_EPSILON"]
        self.epsilon_decay_rate = config["hyperparameters"]["DDDQN"][
            "EPSILON_DECAY_RATE"
        ]

        # Counter for exponential decay
        self.current_episode = 0

        # Model
        self.hidden_features = config["hyperparameters"]["general"][
            "HIDDEN_FEATURES"
            ]
        self.dropout = config["hyperparameters"]["general"]["DROPOUT"]
        self.num_hidden_layers = config["hyperparameters"]["general"][
            "NUM_HIDDEN_LAYERS"
        ]
        self.gamma = config["hyperparameters"]["general"]["GAMMA"]


class DDDQNAgent(BaseQAgent):
    def __init__(
        self,
        env,
        agent_type,
        device,
        loss_type,
        scheduler_type,
    ):
        super().__init__(
            env=env,
            agent_type=agent_type,
            device=device,
            loss_type=loss_type
        )
        # Memory and model
        self.memory = ReplayMemory(config["hyperparameters"]["DDDQN"]["MEMORY_SIZE"])
        self.batch_size = config["hyperparameters"]["general"]["BATCH_SIZE"]

        # Optimizer
        self.scheduler_type = scheduler_type
        self.lr = config["hyperparameters"]["optimizer"]["SCHEDULER_LEARNING_RATE"]
        self.weight_decay = config["hyperparameters"]["optimizer"]["WEIGHT_DECAY"]
        self.exp_decay = config["hyperparameters"]["optimizer"]["EXP_LR_DECAY"]

        # Model
        self.model = DDDQN(
            state_size=self.env.input_features,
            action_size=self.env.action_size,
            hidden_features=self.hidden_features,
            dropout=self.dropout,
            layer_norm=True,
            num_hidden_layers=self.num_hidden_layers,
        ).to(self.device)

        # AdamW Optimizer
        self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.lr,
                amsgrad=True,
                weight_decay=self.weight_decay,
            )
        self.grad_clip = False

        # Learning rate scheduler
        self.scheduler_type = scheduler_type
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

        # Update the target model
        self.target_model = deepcopy(self.model).to(self.device)
        self._update_target_model()

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.env.action_size)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals = self.model(state)
            return torch.argmax(q_vals).item()

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
                self.remember(
                    state,
                    action,
                    step_reward,
                    next_state,
                    done
                    )
                # Update current state
                state = next_state
                # Stop when the episode ends
                if done:
                    break

        # Sample from replay memory
        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.memory.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute loss
        q_vals = self.model(states).gather(
                1,
                actions.unsqueeze(1)
                ).squeeze(1)
        with torch.no_grad():
            next_q_vals = self.target_model(next_states).max(1)[0]
            target_q_vals = rewards + self.gamma * next_q_vals * (1 - dones)

        loss = self.loss(q_vals, target_q_vals)

        # Zero gradients
        self.optimizer.zero_grad(set_to_none=True)

        # Backpropagate
        if self.scaler is not None:
            if self.scheduler_type is not None:
                self.scaler.scale(loss).backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                old_scaler = self.scaler.get_scale()
                self.scaler.update()
                new_scaler = self.scaler.get_scale()
                if old_scaler <= new_scaler:
                    self.lr_scheduler.step()
            else:
                self.scaler.scale(loss).backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        # Update epsilon
        self.epsilon = self.epsilon_min + (
            (self.epsilon_max - self.epsilon_min)
            * math.exp(-self.epsilon_decay_rate * self.current_episode)
        )

        # Update episode
        self.current_episode += 1

        return loss.item()

    def _update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
