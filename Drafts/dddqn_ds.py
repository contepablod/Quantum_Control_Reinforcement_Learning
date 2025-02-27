import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Hyperparameters (Hadamard Gate Example)
state_size = 8  # Real + Imag parts of 2x2 unitary matrix
action_size = 2  # ε ∈ {+4, -4}
gamma = 0.95  # Discount factor
epsilon_start = 1.0  # Initial exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 72  # From paper
learning_rate = 0.001
target_update = 100  # Update target network every 100 steps
memory_capacity = 100000
episodes = 50000  # Training episodes
N = 28  # Time steps (for T=1.0)
T = 1.0
delta_t = T / N

# Pauli Matrices (Complex dtype)
sigma_z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex64)
sigma_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex64)

# Target Unitary (Hadamard)
U_f = torch.tensor(
    [[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]],
    dtype=torch.complex64,
)
D = 2  # Hilbert space dimension


# Dueling Double DQN Network
class DDDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDDQN, self).__init__()
        # Encoder: 3 layers
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU(),
        )
        # Value Stream: 4 layers
        self.value = nn.Sequential(
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, 1),
        )
        # Advantage Stream: 4 layers
        self.advantage = nn.Sequential(
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, action_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        value = self.value(x)
        adv = self.advantage(x)
        return value + (adv - adv.mean(dim=1, keepdim=True))


# Networks and Optimizer
q_net = DDDQN(state_size, action_size)
target_net = DDDQN(state_size, action_size)
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


buffer = ReplayBuffer(memory_capacity)


# Convert Unitary to State Vector
def unitary_to_state(U):
    real = U.real.flatten()
    imag = U.imag.flatten()
    return torch.cat([real, imag]).detach().numpy()


# Training Loop
epsilon = epsilon_start
rewards_log = []

for episode in range(episodes):
    U = torch.eye(2, dtype=torch.complex64)  # Initial state
    state = unitary_to_state(U)
    total_reward = 0

    for t in range(N):
        # Epsilon-greedy action
        if random.random() < epsilon:
            action = random.randint(0, action_size - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_vals = q_net(state_tensor)
            action = torch.argmax(q_vals).item()

        # Get ε value (+4 or -4)
        epsilon_val = 4.0 if action == 0 else -4.0

        # Compute Hamiltonian and unitary evolution
        H = sigma_z + epsilon_val * sigma_x
        U_step = torch.matrix_exp(-1j * H * delta_t)
        U_next = U_step @ U

        # Calculate reward (only at final step)
        if t == N - 1:
            product = torch.conj(U_f).T @ U_next
            trace = torch.trace(product)
            fidelity = (torch.abs(trace) / D) ** 2
            reward = -np.log10(1 - fidelity.item())
            done = True
        else:
            reward = 0
            done = False

        # Store transition
        next_state = unitary_to_state(U_next)
        buffer.push(state, action, reward, next_state, done)

        # Update state
        state = next_state
        U = U_next
        total_reward += reward

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Train network if enough samples
    if len(buffer) >= batch_size:
        batch = buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Current Q-values
        q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q-values
        with torch.no_grad():
            next_q = target_net(next_states).max(1)[0]
            targets = rewards + gamma * next_q * (1 - dones)

        # Update network
        loss = nn.MSELoss()(q_values, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update target network
    if episode % target_update == 0:
        target_net.load_state_dict(q_net.state_dict())

    rewards_log.append(total_reward)
    if episode % 100 == 0:
        print(f"Episode {episode}, Reward: {total_reward:.4f}, Epsilon: {epsilon:.3f}")

# Save model or plot rewards as needed
