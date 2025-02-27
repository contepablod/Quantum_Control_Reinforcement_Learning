import numpy as np
from torch import device
from torch.cuda import is_available

config = {
    "hyperparameters": {
        "general": {
            "HIDDEN_FEATURES": 128,
            "NUM_HIDDEN_LAYERS": 4,
            "DROPOUT": 0.1,
            "BATCH_SIZE": 256,
            "GAMMA": 0.95,
            "TOTAL_TIME": 1,
            "MAX_STEPS": 128,
        },
        "optimizer": {
            "OPTIMIZER_TYPE": "AdamW",
            "SCHEDULER_TYPE": None,  # OR OS
            "SCHEDULER_LEARNING_RATE": 0.001,
            "SCHEDULER_LR_MIN": 1e-6,
            "EXP_LR_DECAY": 0.9999,
            "COS_WARMUP_STEPS": 10000,
            "COS_WARMUP_FACTOR": 5,
            "COS_T_MAX": 1e3,
            "WEIGHT_DECAY": 1e-3,
        },
        "loss": {
            "LOSS_TYPE": "MSE",  # MSE/HUBER
            "HUBER_DELTA_MAX": 10.0,
            "HUBER_DELTA_MIN": 1.0,
            "HUBER_DELTA_DECAY_RATE": 1e-4,
        },
        "train": {
            "EPISODES": 200000,
            "PATIENCE": 30000,
            "FIDELITY_THRESHOLD": 0.999,
            "WINDOW_SIZE": 100,
        },
        "DDDQN": {
            "MEMORY_SIZE": 100000,
            "MAX_EPSILON": 1.0,
            "MIN_EPSILON": 1e-10,
            "EPSILON_DECAY_RATE": 1e-3,
            "TARGET_UPDATE": 100,
        },
        "PPO": {
            "CLIP_EPSILON": 0.2,
            "ENTROPY_COEFF": 0.01,
            "VALUE_COEFF": 0.5,
            "LAMBDA": 0.95,
            "EPOCHS_PPO": 4,
            "TIMESTEPS": 512,
            "BETA": 0.01,
        },
        "TD3": {
            "NOISE": 0.1,
            "POLICY_NOISE": 0.2,
            "TAU": 0.005,
            "NOISE_CLIP": 0.5,
        },
    },
    "pauli_matrices": {
        "I_GATE": np.eye(2, dtype=complex),
        "X_GATE": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y_GATE": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z_GATE": np.array([[1, 0], [0, -1]], dtype=complex),
    },
    "quantum_gates": {
        "H": {
            "num_qubits": 1,
            "unitary": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
        },
        "T": {
            "num_qubits": 1,
            "unitary": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]],
                                dtype=complex),
        },
        "CNOT": {
            "num_qubits": 2,
            "unitary": np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                dtype=complex),
        },
    },
    "device": device("cuda" if is_available() else "cpu"),
    "paths": {
        "MODELS": "/home/pdconte/Desktop/DUTh_Thesis/Models",
        "PLOTS": "/home/pdconte/Desktop/DUTh_Thesis/Plots",
        "DATA": "/home/pdconte/Desktop/DUTh_Thesis/Data",
        "RUNS": "/home/pdconte/Desktop/DUTh_Thesis/Package/runs",
        "METRICS_FORMAT": "csv",
    },
}
