import numpy as np
import torch


config = {
    "hyperparameters": {
        "HIDDEN_FEATURES": 512,
        "DROPOUT": 0.1,
        "BATCH_SIZE": 64,
        "SCHEDULER_TYPE": "EXP",  # OR COS
        "SCHEDULER_LEARNING_RATE": 1e-2,
        "SCHEDULER_LR_MIN": 1e-6,
        "EXP_LR_DECAY": 0.9999,
        "COS_WARMUP_STEPS": 10000,
        "COS_WARMUP_FACTOR": 5,
        "COS_T_MAX": 1e3,
        "WEIGHT_DECAY": 1e-3,
        "MAX_EPSILON": 1.0,
        "MIN_EPSILON": 1e-4,
        "EPSILON_DECAY_RATE": 1e-3,
        "LOSS_TYPE": "HUBER_DYNAMIC",  # OR HUBER_STATIC OR MSE
        "HUBER_DELTA_MAX": 10.0,
        "HUBER_DELTA_MIN": 1.0,
        "HUBER_DELTA_DECAY_RATE": 1e-3,
        "GAMMA": 0.95,
        "MEMORY_SIZE": 100000,
        "TARGET_UPDATE": 100,
        "EPISODES": 200000,
        "PATIENCE": int(200000 / 2),
        "MAX_STEPS": 100,
        "INFIDELITY_THRESHOLD": 5,
        "FIDELITY_DROP_PENALTY_WEIGHT": 0.1,
        "FIDELITY_AVERAGE_WEIGHT": 0.1,
        "FIDELITY_VARIANCE_WEIGHT": 0.1,
    },
    "pauli_matrices": {
        "I_GATE": np.eye(2),
        "X_GATE": np.array([[0, 1], [1, 0]]),
        "Y_GATE": np.array([[0, -1j], [1j, 0]]),
        "Z_GATE": np.array([[1, 0], [0, -1]]),
    },
    "quantum_gates": {
        "HADAMARD_GATE": np.array([[1, 1], [1, -1]],
                                  dtype=np.complex128) / np.sqrt(2),
        "T_GATE": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]],
                           dtype=np.complex128),
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
        "RUNS": "/home/pdconte/Desktop/DUTh_Thesis/Package/runs",
    },
}
