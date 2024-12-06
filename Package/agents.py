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
from utils import DynamicHuberLoss


class BaseQAgent:
    """
    Base class for DQRL agents.
    Implements common functionalities like replay memory, epsilon decay,
    and training logic.
    """

    def __init__(
            self,
            env,
            state_size,
            action_size,
            agent_type,
            loss_type,
            scheduler_type,
            device
            ):

        # Map agent types to model classes
        agent_type_map = {
            "DQN": DQN,
            "DDDQN": DDDQN,
        }

        # Validate agent type
        if agent_type not in agent_type_map:
            raise ValueError(
                f"Invalid agent_type '{agent_type}'. Choose from {list(
                    agent_type_map.keys())}."
            )

        # Environment
        self.env = env

        # Hyperparameters
        self.device = device
        self.state_size = state_size
        self.action_size = action_size

        model_class = agent_type_map[agent_type]

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
        self.loss_type = loss_type
        if loss_type == "MSE":
            self.loss = nn.MSELoss().to(self.device)
        elif loss_type == "HUBER_DYNAMIC":
            # Delta decay
            self.delta = config["hyperparameters"]["HUBER_DELTA_MAX"]
            self.delta_min = config["hyperparameters"]["HUBER_DELTA_MIN"]
            self.delta_decay_rate = \
                config["hyperparameters"]["HUBER_DELTA_DECAY_RATE"]
            self.loss = DynamicHuberLoss(
                delta_initial=self.delta,
                delta_min=self.delta_min,
                delta_decay_rate=self.delta_decay_rate,
            ).to(self.device)
        elif self.loss_type == "HUBER_STATIC":
            self.delta = config["hyperparameters"]["HUBER_DELTA_MAX"]
            self.loss = nn.HuberLoss(
                delta=self.delta
                ).to(self.device)

        # Gradient scaler
        self.scaler = GradScaler(device=self.device.type)

        # Learning rate scheduler
        if scheduler_type == "EXP":
            self.lr_scheduler = ExponentialLR(
                self.optimizer,
                gamma=config["hyperparameters"]["EXP_LR_DECAY"]
            )
        elif scheduler_type == "COS":
            self.lr_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config["hyperparameters"]["COS_WARMUP_STEPS"],
                T_mult=config["hyperparameters"]["COS_WARMUP_FACTOR"],
                eta_min=config["hyperparameters"]["SCHEDULER_LR_MIN"],
            )

        # Epsilon decay
        self.epsilon = config["hyperparameters"]["MAX_EPSILON"]
        self.epsilon_min = config["hyperparameters"]["MIN_EPSILON"]
        self.epsilon_decay_rate = \
            config["hyperparameters"]["EPSILON_DECAY_RATE"]

        # Counter for exponential decay and loss
        self.current_episode = 0

        # Update the target model
        self.update_target_model()

    def update_target_model(self):
        """
        Updates the target model by loading the state dictionary from the
        current model.

        This is used to periodically update the target model during training
        to improve stability of the training process.

        Returns:
        -------
        None
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        """
        Selects an action for the given state using an epsilon-greedy policy.

        Parameters:
        ----------
        state : np.ndarray
            The current state based on which the action is decided.

        Returns:
        -------
        int
            The action selected by the agent.
        """

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.inference_mode():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals = self.model(state)
        return torch.argmax(q_vals).item()

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience in the replay memory.

        Parameters:
        ----------
        state : np.ndarray
            The state observed before taking the action.
        action : int
            The action taken by the agent.
        reward : float
            The reward received after taking the action.
        next_state : np.ndarray
            The state observed after taking the action.
        done : bool
            Whether the episode has ended.
        """
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        """
        Trains the agent on a batch of experiences sampled
        from the replay memory.

        The agent samples a batch of experiences from the replay memory,
        computes the loss using the target network and the current network,
        and backpropagates the error to update the network weights.
        The epsilon value is decayed after each call to this function.

        Returns:
        -------
        float
            The loss of the current batch.
        """
        # Ensure replay memory has enough samples
        while len(self.memory) < config["hyperparameters"]["BATCH_SIZE"]:
            state = self.env.reset()
            for _ in range(self.env.max_steps):
                # Generate a random action to explore
                action = random.randrange(self.action_size)
                (
                    done,
                    next_state,
                    discounted_reward,
                    fidelity,
                    log_infidelity
                ) = self.env.step(action)

                # Add experience to memory
                self.remember(
                    state,
                    action,
                    discounted_reward,
                    next_state,
                    done
                    )
                # Update current state
                state = next_state
                # Stop when the episode ends
                if done:
                    break

        # Sample from replay memory
        batch = self.memory.sample(config["hyperparameters"]["BATCH_SIZE"])
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
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
            loss, delta = self.loss(
                q_vals, target_q_vals, self.current_episode
                ) if self.loss_type == "HUBER_DYNAMIC" else self.loss(
                q_vals, target_q_vals
            )

        # Backpropagate
        self.scaler.scale(loss).backward()
        # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        old_scaler = self.scaler.get_scale()
        self.scaler.update()
        new_scaler = self.scaler.get_scale()

        if old_scaler <= new_scaler:
            self.lr_scheduler.step()

        # Update epsilon
        self.epsilon = self.epsilon_min + (
            (config["hyperparameters"]["MAX_EPSILON"] - self.epsilon_min)
            * math.exp(-self.epsilon_decay_rate * self.current_episode)
        )

        # Store delta
        self.delta = delta

        # Update episode
        self.current_episode += 1

        empty_cache()

        return loss.item()


class DQNAgent(BaseQAgent):
    """
    DQN Agent that inherits shared logic from BaseQAgent.
    """

    def __init__(
            self,
            env,
            state_size,
            action_size,
            agent_type,
            loss_type,
            scheduler_type,
            device
            ):
        super().__init__(
            env=env,
            state_size=state_size,
            action_size=action_size,
            agent_type=agent_type,
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
            env,
            state_size,
            action_size,
            agent_type,
            loss_type,
            scheduler_type,
            device
            ):
        super().__init__(
            env=env,
            state_size=state_size,
            action_size=action_size,
            agent_type=agent_type,
            loss_type=loss_type,
            scheduler_type=scheduler_type,
            device=device,
        )
