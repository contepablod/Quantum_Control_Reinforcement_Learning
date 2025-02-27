import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.tanh(self.fc3(x))


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        return (
            torch.stack(state),
            torch.stack(action),
            torch.tensor(reward).unsqueeze(1),
            torch.stack(next_state),
        )


# TD3 Algorithm
class TD3Agent:
    def __init__(
        self, state_dim, action_dim, hidden_dim, lr, gamma, tau, sigma, buffer_size
    ):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic1 = Critic(state_dim, action_dim, hidden_dim)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim)
        self.critic1_target = Critic(state_dim, action_dim, hidden_dim)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.sigma = sigma

        self._soft_update(self.actor_target, self.actor, 1.0)
        self._soft_update(self.critic1_target, self.critic1, 1.0)
        self._soft_update(self.critic2_target, self.critic2, 1.0)

    def select_action(self, state, noise=0.1):
        state = torch.tensor(state, dtype=torch.float32)
        action = self.actor(state).detach().numpy()
        action += noise * np.random.normal(0, self.sigma, size=action.shape)
        return np.clip(action, -1, 1)

    def train(self, batch_size):
        if len(self.buffer.buffer) < batch_size:
            return

        states, actions, rewards, next_states = self.buffer.sample(batch_size)

        # Compute target Q value
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.sigma).clamp(-0.5, 0.5)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1, 1)
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = rewards + self.gamma * torch.min(target_q1, target_q2)

        # Update critics
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed policy updates
        if np.random.randint(2) == 0:
            # Compute actor loss
            actor_loss = -self.critic1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            self._soft_update(self.actor_target, self.actor, self.tau)
            self._soft_update(self.critic1_target, self.critic1, self.tau)
            self._soft_update(self.critic2_target, self.critic2, self.tau)

    def _soft_update(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1 - tau) * target_param.data
            )


# Example usage
if __name__ == "__main__":
    state_dim = 4  # e.g., [Re(U), Im(U)]
    action_dim = 1  # Quantum control parameter
    hidden_dim = 64
    lr = 1e-3
    gamma = 0.99
    tau = 0.005
    sigma = 0.1
    buffer_size = 100000
    batch_size = 64

    agent = TD3Agent(
        state_dim, action_dim, hidden_dim, lr, gamma, tau, sigma, buffer_size
    )

    # Mock environment for quantum control (replace with actual quantum simulation)
    for episode in range(1000):
        state = np.random.randn(state_dim)
        for step in range(200):
            action = agent.select_action(state)
            next_state = np.random.randn(state_dim)  # Replace with quantum dynamics
            reward = -np.random.rand()  # Replace with infidelity calculation
            agent.buffer.add(
                (
                    torch.tensor(state, dtype=torch.float32),
                    torch.tensor(action, dtype=torch.float32),
                    reward,
                    torch.tensor(next_state, dtype=torch.float32),
                )
            )
            state = next_state
            agent.train(batch_size)
