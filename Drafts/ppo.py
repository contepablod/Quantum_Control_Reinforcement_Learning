import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

# Hyperparameters
GAMMA = 0.99
LAMBDA = 0.95  # GAE Lambda
EPS_CLIP = 0.2  # PPO Clipping Parameter
LR = 3e-4
K_EPOCHS = 10
BATCH_SIZE = 64
ENTROPY_BETA = 0.01


class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.policy_layer = nn.Linear(64, output_dim)
        self.value_layer = nn.Linear(64, 1)

    def forward(self, x):
        shared = self.shared_layers(x)
        policy_logits = self.policy_layer(shared)
        value = self.value_layer(shared)
        return policy_logits, value

    def get_action(self, state):
        state = np.array(state, dtype=np.float32)  # ✅ Convert list to NumPy array first
        state = torch.from_numpy(state)
        logits, _ = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


class PPO:
    def __init__(self, input_dim, output_dim):
        self.actor_critic = ActorCritic(input_dim, output_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=LR)
        self.memory = []

    def store_transition(self, transition):
        self.memory.append(transition)

    def train(self):
        # Convert memory to tensors
        states, actions, old_log_probs, rewards, dones, next_states = zip(*self.memory)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)

        # Compute returns using GAE
        _, next_value = self.actor_critic(next_states[-1])
        advantages, returns = self.compute_gae(rewards, dones, states, next_value)

        for _ in range(K_EPOCHS):
            logits, values = self.actor_critic(states)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)

            # Compute ratios for PPO objective
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (MSE)
            value_loss = 0.5 * (returns - values).pow(2).mean()

            # Entropy bonus for exploration
            entropy = dist.entropy().mean()
            loss = policy_loss + value_loss - ENTROPY_BETA * entropy

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = []

    def compute_gae(self, rewards, dones, states, next_value):
        _, values = self.actor_critic(states)
        values = values.squeeze()

        returns = []
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * next_value * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            next_value = values[t]
        returns = advantages + values.tolist()
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(
            returns, dtype=torch.float32
        )


def train_ppo(env_name="CartPole-v1", episodes=1000):
    env = gym.make(env_name)
    ppo = PPO(env.observation_space.shape[0], env.action_space.n)

    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0

        while not done:
            action, log_prob = ppo.actor_critic.get_action(state)
            next_state, reward, done, _ = env.step(action)

            ppo.store_transition((state, action, log_prob, reward, done, next_state))
            state = next_state
            total_reward += reward

        ppo.train()
        print(f"Episode {episode}, Reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    train_ppo()
