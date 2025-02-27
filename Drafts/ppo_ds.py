import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters (Hadamard Gate Example)
state_size = 8  # Real + Imag parts of 2x2 unitary matrix
action_size = 2  # ε ∈ {+4, -4}
hidden_size = 256  # Hidden layer size
gamma = 0.99  # Discount factor
gae_lambda = 0.95  # GAE parameter
clip_epsilon = 0.2  # PPO clip parameter
entropy_coef = 0.01  # Entropy regularization
epochs = 4  # Training epochs per batch
batch_size = 64  # Minibatch size
lr = 3e-4  # Learning rate
max_episodes = 10000  # Training episodes
N = 30  # Time steps (for T=1.0)
delta_t = 1.0 / N  # Time step duration

# Pauli Matrices (Complex dtype) on GPU
sigma_z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex64, device=device)
sigma_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex64, device=device)

# Target Unitary (Hadamard) on GPU
U_f = torch.tensor(
    [[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]],
    dtype=torch.complex64,
    device=device,
)
D = 2  # Hilbert space dimension


# Convert Unitary to State Vector (with GPU-CPU transfer)
def unitary_to_state(U):
    U_cpu = U.cpu()  # Move to CPU for numpy conversion
    real = U_cpu.real.flatten()
    imag = U_cpu.imag.flatten()
    return torch.cat([real, imag]).detach().numpy()


# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.encoder(x)
        return self.actor(x), self.critic(x)

    def get_action(self, state, action_mask=None):
        state = torch.FloatTensor(state).to(device)  # Move to GPU
        logits, _ = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


# Initialize PPO Agent on GPU
agent = ActorCritic(state_size, action_size, hidden_size).to(device)
optimizer = optim.Adam(agent.parameters(), lr=lr)
fidelity_list = []
reward_list = []

# Training Loop
for episode in range(max_episodes):
    U = torch.eye(2, dtype=torch.complex64, device=device)  # GPU tensor
    state = unitary_to_state(U)
    states, actions, log_probs, rewards, dones = [], [], [], [], []

    # Collect trajectory
    for t in range(N):
        action, log_prob = agent.get_action(state)
        epsilon = 4.0 if action == 0 else -4.0

        # GPU-accelerated matrix operations
        H = sigma_z + epsilon * sigma_x
        U_step = torch.matrix_exp(-1j * H * delta_t)
        U_next = U_step @ U
        next_state = unitary_to_state(U_next)

        if t == N - 1:
            product = torch.conj(U_f).T @ U_next
            trace = torch.trace(product)
            fidelity = (torch.abs(trace) / D) ** 2
            reward = -np.log10(1 - fidelity.item())
            done = True
        else:
            reward = 0.0
            done = False

        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(done)

        state = next_state
        U = U_next

    # Convert buffers to GPU tensors
    states_tensor = torch.FloatTensor(np.array(states)).to(device)
    actions_tensor = torch.LongTensor(actions).to(device)
    old_log_probs_tensor = torch.stack(log_probs).detach().to(device)
    returns_tensor = torch.FloatTensor(rewards).to(device)

    # Normalize returns
    returns_tensor = (returns_tensor - returns_tensor.mean()) / (
        returns_tensor.std() + 1e-8
    )

    # PPO Update with GPU-accelerated batches
    for _ in range(epochs):
        indices = torch.randperm(len(states_tensor)).to(device)
        for idx in indices.split(batch_size):
            batch_states = states_tensor[idx]
            batch_actions = actions_tensor[idx]
            batch_old_log_probs = old_log_probs_tensor[idx]
            batch_returns = returns_tensor[idx]

            logits, values = agent(batch_states)
            new_probs = torch.softmax(logits, dim=-1)
            dist = Categorical(new_probs)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - batch_old_log_probs)
            advantages = batch_returns - values.squeeze().detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (values.squeeze() - batch_returns).pow(2).mean()
            entropy_loss = -entropy_coef * entropy

            loss = policy_loss + value_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

    fidelity_list.append(fidelity.item())
    reward_list.append(reward)

    if episode % 100 == 0:
        print(
            f"Episode {episode}, Final Fidelity: {fidelity.item():.4f}, Reward: {reward:.4f}"
        )

# Plotting remains the same
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(fidelity_list)
plt.title("Fidelity")
plt.subplot(1, 2, 2)
plt.plot(reward_list)
plt.title("Reward")
plt.show()
