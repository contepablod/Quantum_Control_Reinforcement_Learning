import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from copy import deepcopy
import matplotlib.pyplot as plt
import scipy


class DDDQN(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_features,
        dropout,
        layer_norm=False,
        num_hidden_layers=1,
    ):
        super(DDDQN, self).__init__()

        # Shared encoder: num_hidden_layers - 1
        encoder_layers = []
        input_size = state_size
        for _ in range(num_hidden_layers - 1):
            encoder_layers.append(nn.Linear(input_size, hidden_features))
            if layer_norm:
                encoder_layers.append(nn.LayerNorm(hidden_features))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(p=dropout))
            input_size = hidden_features
        self.encoder = nn.Sequential(*encoder_layers)

        # Advantage stream: num_hidden_layers
        advantage_layers = []
        input_size = hidden_features
        for _ in range(num_hidden_layers):
            advantage_layers.append(nn.Linear(input_size, hidden_features))
            advantage_layers.append(nn.ReLU())
            input_size = hidden_features
        advantage_layers.append(nn.Linear(hidden_features, action_size))  # Output layer
        self.advantage_stream = nn.Sequential(*advantage_layers)

        # Value stream: num_hidden_layers
        value_layers = []
        input_size = hidden_features
        for _ in range(num_hidden_layers):
            value_layers.append(nn.Linear(input_size, hidden_features))
            value_layers.append(nn.ReLU())
            input_size = hidden_features
        value_layers.append(nn.Linear(hidden_features, 1))  # Output layer
        self.value_stream = nn.Sequential(*value_layers)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        encoded = self.encoder(x)
        value = self.value_stream(encoded)
        advantage = self.advantage_stream(encoded)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def _initialize_weights(self, init_type="kaiming_uniform"):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                if init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                elif init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(layer.weight)
                else:
                    raise ValueError(f"Unsupported init_type: {init_type}")

                # Set biases to zero
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.apply(init_weights)


# Define the environment dynamics for the quantum control problem
class QuantumEnvironment:
    def __init__(self, target_unitary, action_dim, max_steps):
        self.target_unitary = target_unitary
        self.current_unitary = np.eye(target_unitary.shape[0], dtype=complex)
        self.max_steps = max_steps
        self.time_delta = 1/self.max_steps
        self.control_pulse_params = {
            "omega": np.linspace(-1, 1, action_dim),
            "delta": np.linspace(-1, 1, action_dim)
        }

    def reset(self):
        self.current_unitary = np.eye(self.target_unitary.shape[0], dtype=complex)
        self.time_step = 0
        return self._get_state()

    def step(self, action):
        self.time_step += 1
        num_delta = len(self.control_pulse_params["delta"])

        if not (0 <= action < self.action_size):
            raise ValueError(f"Action {action} out of bounds. Valid range: 0 to {self.action_size - 1}.")

        omega_index = action // num_delta
        delta_index = action % num_delta
        omega = self.control_pulse_params["omega"][omega_index]
        delta = self.control_pulse_params["delta"][delta_index]

        hamiltonian = (omega / 2) * np.array([[0, 1], [1, 0]], dtype=complex) + (delta / 2) * np.array([[1, 0], [0, -1]], dtype=complex)
        evolution = scipy.linalg.expm(-1j * hamiltonian * self.time_delta)
        self.current_unitary = evolution @ self.current_unitary
        reward = 0
        done = False
        # if self._fidelity() > 0.9999:
        if self.time_step >= self.max_steps:
            done = True
            self.fidelity = self._fidelity()
            reward = -np.log10(1 - self.fidelity)
        return self._get_state(), reward, done

    def _get_state(self):
        real_part = np.real(self.current_unitary.ravel())
        imag_part = np.imag(self.current_unitary.ravel())
        return np.concatenate([real_part, imag_part])

    def _fidelity(self):
        overlap = (
            np.trace(self.target_unitary.conj().T @ self.current_unitary)
        )
        return np.abs(overlap / self.target_unitary.shape[0])**2


# Define the replay buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def size(self):
        return len(self.buffer)


# Define the agent
class DDDQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        env,
        learning_rate,
        gamma,
        epsilon_decay,
        replay_buffer_size,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env = env
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 1e-3
        self.epsilon_decay = epsilon_decay
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.eval_net = DDDQN(state_dim, action_dim, hidden_features=256, dropout=0.1, layer_norm=True, num_hidden_layers=4).to(device)
        self.target_net = deepcopy(self.eval_net).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self.eval_net(state_tensor)
        return torch.argmax(q_values).item()

    def update(self, batch_size):
        while self.replay_buffer.size() < batch_size:
            state = self.env.reset()
            for _ in range(self.env.max_steps):
                # Generate a random action to explore
                action = agent.select_action(state)
                (
                    next_state,
                    reward,
                    done
                ) = self.env.step(action)
                # Add experience to memory
                self.replay_buffer.add((state, action, reward, next_state, done))
                # Update current state
                state = next_state
                # Stop when the episode ends
                if done:
                    break

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        current_q = self.eval_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())


# Training loop
def train(agent, env, num_episodes, batch_size, update_target_period):

    fidelities = []
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        c = 0
        loss = 0.0
        # while not done:
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.replay_buffer.add((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            loss += agent.update(batch_size)
            c += 1
            if done:
                break
        avg_loss = loss / c

        fidelities.append(env.fidelity)
        rewards.append(total_reward)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        print(f"Episode {episode}, Total Reward: {total_reward}, Fidelity: {env.fidelity}, Epsilon: {agent.epsilon}, Counter = {c}, Loss = {avg_loss}")
        if episode % update_target_period == 0:
            agent.update_target_network()

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Evolution")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(fidelities, label="Fidelity")
    plt.xlabel("Episode")
    plt.ylabel("Fidelity")
    plt.title("Fidelity Evolution")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Hyperparameters
state_dim = 8  # Example size
action_dim = 10  # Example size
learning_rate = 0.001
gamma = 0.99
epsilon_decay = 0.99
replay_buffer_size = 100000
num_episodes = 10000
batch_size = 64
update_target_period = 10
max_steps = 50

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment and Agent initialization
# Replace with the proper Hamiltonians and target unitaries
target_unitary = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # Example Hadamard gate
env = QuantumEnvironment(
    target_unitary, action_dim, max_steps)

agent = DDDQNAgent(
    state_dim, action_dim, env, learning_rate, gamma, epsilon_decay, replay_buffer_size
)
torch.cuda.empty_cache()
# Train the agent
train(agent, env, num_episodes, batch_size, update_target_period)
