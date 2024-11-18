import numpy as np
import torch


config = {
    "hyperparameters": {
        "HIDDEN_FEATURES": 512,
        "LEARNING_RATE": 1e-4,
        "WEIGHT_DECAY": 1e-3,
        "DROPOUT": 0.1,
        "GAMMA": 0.9,
        "EPSILON": 1.0,
        "EPSILON_DECAY": 0.9999,
        "MIN_EPSILON": 0.0001,
        "MEMORY_SIZE": 10000,
        "BATCH_SIZE": 512,
        "TARGET_UPDATE": 10,
        "EPISODES": 150000,
        "PATIENCE": 150000 / 2,
        "MAX_STEPS": 10,
        "FIDELITY_THRESHOLD": 1e-5,
    },
    "pauli_matrices": {
        "I_GATE": np.eye(2),
        "X_GATE": np.array([[0, 1], [1, 0]]),
        "Y_GATE": np.array([[0, -1j], [1j, 0]]),
        "Z_GATE": np.array([[1, 0], [0, -1]]),
    },
    "hamiltonian": {
        "OMEGA": 1.0,
        "J": 0.1,
    },
    "quantum_gates": {
        "HADAMARD_GATE": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
        "T_GATE": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128),
        "CNOT_GATE": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
    },
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}