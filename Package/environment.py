import numpy as np
import random
from fidelities import state_fidelity
from hyperparameters import config
from scipy.linalg import expm


class QuantumGateEnv:
    """
    A class to represent the environment for quantum gate control
    using reinforcement learning.

    Attributes:
    ----------
    gate : str
        The type of quantum gate (e.g., 'H', 'T', 'CNOT').
    control_pulse_params : dict
        Dictionary containing amplitude, phase, and duration parameters
        for control pulses.
    initial_state : np.ndarray
        The initial quantum state of the system.
    state : np.ndarray
        The current quantum state of the system.
    target : np.ndarray
        The target quantum gate matrix.
    theoretical_state : np.ndarray
        The expected quantum state after applying the target gate.
    time_step : int
        The current time step in the episode.
    max_steps : int
        The maximum number of steps in an episode.
    state_history : list
        A history of quantum states during an episode.
    """

    def __init__(self, gate):
        """
        Initializes the QuantumGateEnv with the specified gate type.

        Parameters:
        ----------
        gate : str
            The type of quantum gate (e.g., 'H', 'T', 'CNOT').
        """
        self.control_pulse_params = {
            "amplitude": np.linspace(0, 1, 10),  # Example amplitudes
            "phase": np.linspace(0, 2 * np.pi, 12),  # Example phases
            "duration": np.linspace(0.1, 1, 10),  # Example durations
        }
        self.gate = gate
        (
            self.initial_state,
            self.target_state,
            self.state_size,
            self.action_size,
            self.input_features,
        ) = self.set_quantum_info()
        self.reset()

    def _generate_state(self, num_qubits, basis="random"):
        """
        Generate a quantum state for a given number of qubits.

        Parameters:
        - num_qubits (int): The number of qubits.
        - basis (str): The type of state to generate. Options are:
            - "random": A random state in the full Hilbert space.
            - "Z": A state in the Z (computational) basis.
            - "X": A state in the Hadamard (X) basis.
            - "Y": A state in the Y basis.

        Returns:
        - state (np.ndarray): A strictly normalized complex numpy array representing the state vector.
        """
        dim = 2**num_qubits

        if basis == "random":
            # Generate random complex amplitudes
            real_part = np.random.normal(size=dim)
            imag_part = np.random.normal(size=dim)
            state_vector = real_part + 1j * imag_part

        elif basis == "Z":
            # Generate a state in the Z basis (one-hot encoded vector)
            index = np.random.randint(0, dim)
            state_vector = np.zeros(dim, dtype=np.complex128)
            state_vector[index] = 1.0

        elif basis == "X":
            # Generate a state in the Hadamard (X) basis
            state_vector = np.zeros(dim, dtype=np.complex128)
            index = np.random.randint(0, dim)
            if index % 2 == 0:
                # State |+> = (|0> + |1>) / sqrt(2)
                state_vector[index] = 1 / np.sqrt(2)
                state_vector[index + 1] = 1 / np.sqrt(2)
            else:
                # State |-> = (|0> - |1>) / sqrt(2)
                state_vector[index - 1] = 1 / np.sqrt(2)
                state_vector[index] = -1 / np.sqrt(2)

        elif basis == "Y":
            # Generate a state in the Y basis
            state_vector = np.zeros(dim, dtype=np.complex128)
            index = np.random.randint(0, dim)
            if index % 2 == 0:
                # State (|0> + i|1>) / sqrt(2)
                state_vector[index] = 1 / np.sqrt(2)
                state_vector[index + 1] = 1j / np.sqrt(2)
            else:
                # State (|0> - i|1>) / sqrt(2)
                state_vector[index - 1] = 1 / np.sqrt(2)
                state_vector[index] = -1j / np.sqrt(2)

        else:
            raise ValueError("Invalid basis option. Choose 'random', 'Z', 'X', or 'Y'.")

        # Strict normalization to correct any floating-point errors
        norm = np.linalg.norm(state_vector)
        state = state_vector / norm

        return state

    def set_quantum_info(self):
        """
        Sets the quantum information for the environment based on the gate type.

        Returns:
        -------
        tuple
            A tuple containing the initial state, target_state, state size,
            action size, and input features.
        """
        action_size = (
            len(self.control_pulse_params["amplitude"])
            * len(self.control_pulse_params["phase"])
            * len(self.control_pulse_params["duration"])
        )

        if self.gate in ["H", "T"]:
            num_qubits = 1
            state_size = 2**num_qubits
            initial_state = self._generate_state(num_qubits, "random")
            input_features = 2 ** (
                num_qubits + 1
            )  # Number of states in the input space
            input_features = 2 ** (
                num_qubits + 1
            )  # Number of states in the input space
            # target_unitary = self.target_unitary
            unitary = config["quantum_gates"]["HADAMARD_GATE"] if self.gate == "H" else config["quantum_gates"]["T_GATE"]
            target_state = np.dot(initial_state, unitary)

        elif self.gate == "CNOT":
            num_qubits = 2
            state_size = 2**num_qubits
            initial_states = [
                np.array([1, 0, 0, 0], dtype=np.complex128),
                np.array([0, 1, 0, 0], dtype=np.complex128),
                np.array([0, 0, 1, 0], dtype=np.complex128),
                np.array([0, 0, 0, 1], dtype=np.complex128),
            ]
            initial_state = random.choice(initial_states)
            input_features = 2 ** (
                num_qubits + 1
            )  # Number of states in the input space
            # target_unitary = self.target_unitary
            unitary = config["quantum_gates"]["CNOT_GATE"]
            target_state = np.dot(initial_state, unitary)

        return (
            initial_state,
            target_state,
            state_size,
            action_size,
            input_features,
        )

    def reset(self):
        """
        Resets the environment to the initial state.

        Returns:
        -------
        np.ndarray
            The initial quantum state.
        """
        self.state = self.initial_state.copy()
        self.time_step = 0
        self.max_steps = config["hyperparameters"]["MAX_STEPS"]
        self.reward_episode = [-1]
        self.fidelity_episode = [0]
        self.state_episode = [self.state]
        self.amplitude_episode = [0]
        self.phase_episode = [0]
        self.duration_episode = [0]
        return self.state

    def _time_evolution_operator(self, H, t):
        """
        Compute the time evolution operator U(t) = exp(-i * H * t).

        Parameters:
        H (ndarray): The Hamiltonian matrix.
        t (float): The time over which to evolve the system.

        Returns:
        ndarray: The time evolution operator U(t).
        """
        return expm(-1j * H * t)

    def _construct_hamiltonian(self, amplitude, phase):
        """
        Constructs the Hamiltonian for the quantum gate operation based on the gate type.

        Parameters:
        ----------
        amplitude : float
            The amplitude of the control Hamiltonian, which modulates the strength of the control fields.
        phase : float
            The phase of the control Hamiltonian, which determines the direction of the control field in the XY-plane.

        Returns:
        -------
        H_total : ndarray
            The total Hamiltonian (system + control) for the specified gate operation.
            If the gate is "H" or "T", returns a 2x2 matrix.
            If the gate is "CNOT", returns a 4x4 matrix.
        """
        if self.gate in ["H", "T"]:
            H_sys = (
                config["hamiltonian"]["OMEGA"] / 2
            ) * config["pauli_matrices"]["Z_GATE"]  # System Hamiltonian (static)
            H_control = amplitude * (
                np.cos(phase) * config["pauli_matrices"]["X_GATE"]
                + np.sin(phase) * config["pauli_matrices"]["Y_GATE"]
            )  # Control Hamiltonian (depends on amplitude and phase)
            return H_sys + H_control  # Total Hamiltonian
        
        elif self.gate == "CNOT":
            H_system = (
                (config["hamiltonian"]["OMEGA"] / 2)
                * np.kron(
                    config["pauli_matrices"]["Z_GATE"],
                    config["pauli_matrices"]["I_GATE"],
                )
                + (config["hamiltonian"]["OMEGA"] / 2)
                * np.kron(
                    config["pauli_matrices"]["I_GATE"],
                    config["pauli_matrices"]["Z_GATE"],
                )
                + config["hamiltonian"]["J"]
                * (
                    np.kron(
                        config["pauli_matrices"]["X_GATE"],
                        config["pauli_matrices"]["X_GATE"],
                    )
                    + np.kron(
                        config["pauli_matrices"]["Y_GATE"],
                        config["pauli_matrices"]["Y_GATE"],
                    )
                )
            )  # System Hamiltonian (static)
            H_control = amplitude * (
                np.kron(
                    config["pauli_matrices"]["X_GATE"],
                    config["pauli_matrices"]["I_GATE"],
                )
                * np.cos(phase)
                + np.kron(
                    config["pauli_matrices"]["Y_GATE"],
                    config["pauli_matrices"]["I_GATE"],
                )
                * np.sin(phase)
            )  # Control Hamiltonian (depends on amplitude and phase)
            return H_system + H_control  # Total Hamiltonian

    def infidelity(self, final_state):
        """
        Calculates the infidelity between the final state and the theoretical state.

        Parameters:
        ----------
        final_state (np.ndarray): The final quantum state after applying the control pulse.

        Returns:
        -------
        float
            The infidelity between the final state and the theoretical state.
        """
        fidelity = state_fidelity(self.target_state, final_state)
        return 1 - fidelity

    def step(self, action):
        """
        Takes a step in the environment using the given action.

        Parameters:
        ----------
        action : int
        The action to be taken by the agent.

        Returns:
        -------
        tuple
        A tuple containing the next state, reward, and done flag.
        """
        self.time_step += 1

        num_phases = len(self.control_pulse_params["phase"])
        num_durations = len(self.control_pulse_params["duration"])

        amplitude_index = action // (num_phases * num_durations)
        phase_index = (action // num_durations) % num_phases
        duration_index = action % num_durations

        amplitude = self.control_pulse_params["amplitude"][amplitude_index]
        phase = self.control_pulse_params["phase"][phase_index]
        duration = self.control_pulse_params["duration"][duration_index]

        control_hamiltonian = self._construct_hamiltonian(amplitude, phase)
        control_matrix = self._time_evolution_operator(control_hamiltonian, duration)

        next_state = np.dot(control_matrix, self.state)
        self.state = next_state

        # Store episode
        self.state_episode.append(self.state)
        self.amplitude_episode.append(amplitude)
        self.phase_episode.append(phase)
        self.duration_episode.append(duration)

        reward = -self.infidelity(next_state) if self.time_step == self.max_steps else 0
        fidelity = 1 + reward
        self.reward_episode.append(reward)
        self.fidelity_episode.append(fidelity)
        done = self.time_step == self.max_steps

        return (
            done,
            next_state,
            amplitude,
            phase,
            duration,
            reward,
            fidelity,
        )
