import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym

# Define Hyperparameters
GAMMA = 0.99
LAM = 0.95
CLIP_EPS = 0.2
LEARNING_RATE = 3e-4
EPOCHS = 10
BATCH_SIZE = 64
MAX_STEPS = 200  # Maximum steps per episode


# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.actor(x), dim=-1), self.critic(x)


# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        probs, _ = self.policy(state)
        action = torch.multinomial(probs, 1).item()
        return action, probs[action].item()

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * LAM * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, states, actions, log_probs_old, returns, advantages):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        for _ in range(EPOCHS):
            probs, values = self.policy(states)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(log_probs - log_probs_old)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


# Training
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = PPOAgent(state_dim, action_dim)
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

    for step in range(MAX_STEPS):
        action, log_prob = agent.get_action(state.unsqueeze(1))
        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        dones.append(done)

        _, value = agent.policy(torch.tensor(state, dtype=torch.float32))
        values.append(value.item())

        state = next_state

        if done:
            break

    _, next_value = agent.policy(torch.tensor(state, dtype=torch.float32))
    values.append(next_value.item())
    advantages = agent.compute_advantages(rewards, values, dones)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]

    agent.update(states, actions, log_probs, returns, advantages)

    print(f"Episode {episode + 1}: Total Reward = {sum(rewards)}")

env.close()
