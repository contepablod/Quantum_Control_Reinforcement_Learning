import numpy as np
import torch


config = {
    "hyperparameters": {
        "HIDDEN_FEATURES": 64,
        "SCHEDULER_LEARNING_RATE": 1e-2,
        "SCHEDULER_WARMUP_STEPS": 10000,
        "SCHEDULER_WARMUP_FACTOR": 5,
        "SCHEDULER_T_MAX": 1e3,
        "SCHEDULER_LR_MIN": 1e-5,
        "WEIGHT_DECAY": 1e-3,
        "GAMMA_LR_DECAY": 0.9999,
        "MAX_EPSILON": 1.0,
        "MIN_EPSILON": 1e-4,
        "EPSILON_DECAY_RATE": 0.0001,
        "DROPOUT": 0.1,
        "GAMMA": 0.9,
        "LOSS_TYPE": "HUBER",
        "SCHEDULER_TYPE": "EXP",
        "MEMORY_SIZE": 10000,
        "BATCH_SIZE": 512,
        "TARGET_UPDATE": 10,
        "EPISODES": 150000,
        "PATIENCE": 150000 / 2,
        "MAX_STEPS": 10,
        "INFIDELITY_THRESHOLD": 5,
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
        "HADAMARD_GATE": np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2),
        "T_GATE": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128),
        "CNOT_GATE": np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=np.complex128,
        ),
    },
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "paths": {
        "MODELS": "./Models/",
        "PLOTS": "/home/pdconte/Desktop/DUTh_Thesis/Plots",
        "DATA": "./Data/",
    },
}
