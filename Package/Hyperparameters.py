# Hyperparameters
LEARNING_RATE = 1e-4
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.0001
MEMORY_SIZE = 10000
BATCH_SIZE = 512
TARGET_UPDATE = 10
EPISODES = 150000
PATIENCE = EPISODES / 2
MAX_STEPS = 10
FIDELITY_THRESHOLD = 1e-5

# Define Pauli matrices
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

# Define Hamiltonian parameters
omega = 1.0
J = 0.1

# Define Gates
HADAMARD = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
T_GATE = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
