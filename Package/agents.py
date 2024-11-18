# DDDQN Agent
import numpy as np
import random
import torch
import torch.nn as nn
from memory import ReplayMemory
from hyperparameters import config
from networks import DDDQN
from torch import autocast, GradScaler
from torch.optim import AdamW


class DDDQNAgent:
    """
    A Dueling Double Deep Q-Network (DDDQN) agent for reinforcement learning
    in quantum gate control.

    Attributes:
    ----------
    state_size : int
        The size of the state space.
    action_size : int
        The size of the action space.
    epsilon : float
        The exploration rate for the epsilon-greedy policy.
    memory : ReplayMemory
        The replay memory to store experiences.
    model : DDDQN
        The Q-network model for learning the Q-values.
    target_model : DDDQN
        The target Q-network model for stable learning.
    optimizer : torch.optim.Adam
        The optimizer for training the model.
    scaler : torch.cuda.amp.GradScaler
        The gradient scaler for mixed precision training.
    loss : torch.nn.MSELoss
        The loss function for training the model.
    """

    def __init__(self, state_size, action_size):
        """
        Initializes the DDDQNAgent with the given state and action sizes.

        Parameters:
        ----------
        state_size : int
            The size of the state space.
        action_size : int
            The size of the action space.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = config["hyperparameters"]["EPSILON"]
        self.memory = ReplayMemory(config["hyperparameters"]["MEMORY_SIZE"])
        self.model = DDDQN(state_size, action_size).to(config["device"])
        self.target_model = DDDQN(state_size, action_size).to(config["device"])
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config["hyperparameters"]["LEARNING_RATE"],
            amsgrad=True,
            weight_decay=config["hyperparameters"]["WEIGHT_DECAY"],
        )
        self.scaler = GradScaler(device=config["device"].type)
        self.update_target_model()
        self.loss = nn.MSELoss().to(config["device"])

    def update_target_model(self):
        """
        Updates the target model by copying the weights from the current model.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        """
        Selects an action using the epsilon-greedy policy.

        Parameters:
        ----------
        state : np.ndarray
            The current state.

        Returns:
        -------
        int
            The action selected by the agent.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = (
            torch.FloatTensor(np.concatenate([state.real, state.imag]))
            .unsqueeze(0)
            .to(config["device"])
        )
        q_vals = self.model(state)
        return torch.argmax(q_vals).item()

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience in the replay memory.

        Parameters:
        ----------
        state : np.ndarray
            The state before taking the action.
        action : int
            The action taken by the agent.
        reward : float
            The reward received after taking the action.
        next_state : np.ndarray
            The state after taking the action.
        done : bool
            Whether the episode has ended.
        """
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        """
        Trains the model by replaying a batch of experiences from the replay memory.
        """
        if len(self.memory) < config["hyperparameters"]["BATCH_SIZE"]:
            return
        batch = self.memory.sample(config["hyperparameters"]["BATCH_SIZE"])
        states, actions, rewards, next_states, dones = zip(*batch)

        self.model.train()
        self.target_model.eval()

        states = torch.FloatTensor(
            np.array([np.concatenate([s.real, s.imag]) for s in states])
        ).to(config["device"])
        actions = torch.LongTensor(actions).to(config["device"])
        rewards = torch.FloatTensor(rewards).to(config["device"])
        next_states = torch.FloatTensor(
            np.array([np.concatenate([s.real, s.imag]) for s in next_states])
        ).to(config["device"])
        dones = torch.FloatTensor(dones).to(config["device"])

        with autocast(device_type=config["device"].type):
            q_vals = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_vals = self.target_model(next_states).max(1)[0]
            target_q_vals = rewards + (config["hyperparameters"]["GAMMA"] * next_q_vals * (1 - dones))
            loss = self.loss(q_vals, target_q_vals)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.epsilon > config["hyperparameters"]["MIN_EPSILON"]:
            self.epsilon *= config["hyperparameters"]["EPSILON_DECAY"]
