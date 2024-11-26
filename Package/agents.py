import math
import numpy as np
import random
import torch
import torch.nn as nn
from memory import ReplayMemory
from hyperparameters import config
from networks import DDDQN, DQN
from torch import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ExponentialLR,
    CosineAnnealingWarmRestarts
)
from torch.cuda import empty_cache


class BaseQAgent:
    """
    Base class for DQRL agents.
    Implements common functionalities like replay memory, epsilon decay,
    and training logic.
    """

    def __init__(
            self,
            state_size,
            action_size,
            model_class,
            loss_type,
            scheduler_type,
            device
            ):
        # Hyperparameters
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        # Memory and model
        self.memory = ReplayMemory(config["hyperparameters"]["MEMORY_SIZE"])
        self.model = model_class(state_size, action_size).to(self.device)
        self.target_model = model_class(state_size, action_size).to(
            self.device
            )
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config["hyperparameters"]["SCHEDULER_LEARNING_RATE"],
            amsgrad=True,
            weight_decay=config["hyperparameters"]["WEIGHT_DECAY"],
        )
        # Loss
        if loss_type == "MSE":
            self.loss = nn.MSELoss().to(self.device)
        elif loss_type == "HUBER":
            self.loss = nn.HuberLoss().to(self.device)
        self.scaler = GradScaler(device=self.device.type)
        # Learning rate scheduler
        if scheduler_type == "EXP":
            self.lr_scheduler = ExponentialLR(
                self.optimizer,
                gamma=config["hyperparameters"]["GAMMA_LR_DECAY"]
            )
        elif scheduler_type == "COS":
            self.lr_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config["hyperparameters"]["SCHEDULER_WARMUP_STEPS"],
                T_mult=config["hyperparameters"]["SCHEDULER_WARMUP_FACTOR"],
                eta_min=config["hyperparameters"]["SCHEDULER_LR_MIN"],
            )
        # Epsilon decay   
        self.epsilon = config["hyperparameters"]["MAX_EPSILON"]
        self.epsilon_min = config["hyperparameters"]["MIN_EPSILON"]
        self.epsilon_decay_rate = config["hyperparameters"]["EPSILON_DECAY_RATE"]
        self.current_episode = 0  # Counter for exponential decay
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = (
            torch.FloatTensor(np.concatenate([state.real, state.imag]))
            .unsqueeze(0)
            .to(self.device)
        )
        q_vals = self.model(state)
        return torch.argmax(q_vals).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < config["hyperparameters"]["BATCH_SIZE"]:
            return None

        batch = self.memory.sample(config["hyperparameters"]["BATCH_SIZE"])
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(
            np.array([np.concatenate([s.real, s.imag]) for s in states])
        ).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(
            np.array([np.concatenate([s.real, s.imag]) for s in next_states])
        ).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Zero gradients
        self.optimizer.zero_grad(set_to_none=True)

        # Compute loss
        with autocast(device_type=self.device.type):
            q_vals = self.model(states).gather(
                1, actions.unsqueeze(1)
                ).squeeze(1)
            next_q_vals = self.target_model(next_states).max(1)[0]
            target_q_vals = rewards + (
                config["hyperparameters"]["GAMMA"] * next_q_vals * (1 - dones)
            )
            loss = self.loss(q_vals, target_q_vals)

        # Backpropagate
        self.scaler.scale(loss).backward()
        #nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update epsilon
        self.epsilon = self.epsilon_min + (
            (config["hyperparameters"]["MAX_EPSILON"] - self.epsilon_min)
            * math.exp(-self.epsilon_decay_rate * self.current_episode)
        )
        self.current_episode += 1

        self.lr_scheduler.step()

        empty_cache()

        return loss.item()


class DQNAgent(BaseQAgent):
    """
    DQN Agent that inherits shared logic from BaseQAgent.
    """

    def __init__(
            self,
            state_size,
            action_size,
            loss_type,
            scheduler_type,
            device
            ):
        super().__init__(
            state_size,
            action_size,
            model_class=DQN,
            loss_type=loss_type,
            scheduler_type=scheduler_type,
            device=device
        )


class DDDQNAgent(BaseQAgent):
    """
    DDDQN Agent that inherits shared logic from BaseQAgent.
    """

    def __init__(
            self,
            state_size,
            action_size,
            loss_type,
            scheduler_type,
            device
            ):
        super().__init__(
            state_size,
            action_size,
            model_class=DDDQN,
            loss_type=loss_type,
            scheduler_type=scheduler_type,
            device=device,
        )
