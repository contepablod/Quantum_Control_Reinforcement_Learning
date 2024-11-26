import numpy as np
import random
from fidelities import state_fidelity, gate_fidelity
from hyperparameters import config
from scipy.linalg import expm


class QuantumEnv:
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

    def __init__(self, gate, fidelity_type, basis_type, hamiltonian_type):
        """
        Initializes the QuantumGateEnv with the specified gate type.

        Parameters:
        ----------
        gate : str
            The type of quantum gate (e.g., 'H', 'T', 'CNOT').
        fidelity_type: str
            The type of fidelity to be used (e.g., 'state', 'gate', map')
        basis_type: str
            The type of basis to be used (e.g., 'Z', 'X', 'Y' or 'random')
        """
        self.control_pulse_params = {
            "amplitude": np.linspace(0, 1, 10),  # Example amplitudes
            "phase": np.linspace(0, 2 * np.pi, 10),  # Example phases
            "duration": np.linspace(0.1, 1, 10),  # Example durations
        }
        self.gate = gate
        self.fidelity_type = fidelity_type
        self.basis_type = basis_type
        self.hamiltonian_type = hamiltonian_type
        (
            self.initial_state,
            self.target_state,
            self.state_size,
            self.action_size,
            self.input_features,
        ) = self.set_quantum_environment()
        self.reset()

    def set_quantum_environment(self):
        """
        Sets the quantum information for the environment based on
        the gate type.

        Returns:
        -------
        tuple
            A tuple containing the initial state, target_state, state size,
            action size, and input features.
        """

        # Map of gates to their configurations
        gate_config = {
            "H": {"num_qubits": 1, "unitary": config["quantum_gates"]["HADAMARD_GATE"]},
            "T": {"num_qubits": 1, "unitary": config["quantum_gates"]["T_GATE"]},
            "CNOT": {"num_qubits": 2, "unitary": config["quantum_gates"]["CNOT_GATE"]},
        }

        if self.gate not in gate_config:
            raise ValueError(f"Unsupported gate: {self.gate}")

        # Get gate-specific configuration
        num_qubits = gate_config[self.gate]["num_qubits"]
        target_unitary = gate_config[self.gate]["unitary"]

        # Compute shared parameters
        state_size = 2**num_qubits
        input_features = 2 ** (num_qubits + 1)  # Input space size
        action_size = (
            len(self.control_pulse_params["amplitude"])
            * len(self.control_pulse_params["phase"])
            * len(self.control_pulse_params["duration"])
        )

        # Generate initial state
        if num_qubits == 1:
            initial_state = self._generate_state(num_qubits)
        elif num_qubits == 2:
            initial_states = [
                np.array([1, 0, 0, 0], dtype=np.complex128),
                np.array([0, 1, 0, 0], dtype=np.complex128),
                np.array([0, 0, 1, 0], dtype=np.complex128),
                np.array([0, 0, 0, 1], dtype=np.complex128),
            ]
            initial_state = random.choice(initial_states)

        # Handle fidelity types
        if self.fidelity_type == "state":
            target_state = np.dot(initial_state, target_unitary)
            return (
                initial_state,
                target_state,
                state_size,
                action_size,
                input_features
            )

        elif self.fidelity_type == "gate":
            return (
                initial_state,
                target_unitary,
                state_size,
                action_size,
                input_features
            )

        else:
            raise ValueError(f"Unsupported fidelity type: \
                             {self.fidelity_type}")

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

    def log_infidelity(self, final_state):
        """
        Calculates the infidelity between the final state and
        the theoretical state.

        Parameters:
        ----------
        final_state (np.ndarray): The final quantum state after applying
        the control pulse.

        Returns:
        -------
        float
            The infidelity between the final state and the theoretical state.
        """
        if self.fidelity_type == "state":
            fidelity = state_fidelity(final_state, self.target_state)
            return -np.log10(1 - fidelity), fidelity
        elif self.fidelity_type == "gate":
            fidelity = gate_fidelity(final_state, self.target_state)
            return -np.log10(1 - fidelity), fidelity

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
        control_matrix = self._time_evolution_operator(
            control_hamiltonian,
            duration
        )

        next_state = np.dot(control_matrix, self.state)
        self.state = next_state

        # Store episode
        self.state_episode.append(self.state)
        self.amplitude_episode.append(amplitude)
        self.phase_episode.append(phase)
        self.duration_episode.append(duration)

        log_infidelity, fidelity = self.log_infidelity(next_state)
        reward = log_infidelity if self.time_step == self.max_steps else 0
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

    def _generate_state(self, num_qubits):
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
        - state (np.ndarray): A strictly normalized complex numpy array
        representing the state vector.
        """
        dim = 2**num_qubits

        if self.basis_type == "random":
            # Generate random complex amplitudes
            real_part = np.random.normal(size=dim)
            imag_part = np.random.normal(size=dim)
            state_vector = real_part + 1j * imag_part

        elif self.basis_type == "Z":
            # Generate a state in the Z basis (one-hot encoded vector)
            index = np.random.randint(0, dim)
            state_vector = np.zeros(dim, dtype=np.complex128)
            state_vector[index] = 1.0

        elif self.basis_type == "X":
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

        elif self.basis_type == "Y":
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
            raise ValueError("Invalid basis option. Choose 'random', 'Z',\
                            'X', or 'Y'.")

        # Strict normalization to correct any floating-point errors
        norm = np.linalg.norm(state_vector)
        state = state_vector / norm

        return state

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
        Constructs the Hamiltonian for the quantum gate operation based
        on the gate type and Hamiltonian type.

        Parameters:
        ----------
        amplitude : float
            The amplitude of the control Hamiltonian, which modulates
            the strength of the control fields.
        phase : float
            The phase of the control Hamiltonian, which determines
            the direction of the control field in the XY-plane.
        hamiltonian_type : str
            The type of Hamiltonian to use ("field_driven", "rotational",
            "geometric", "composite").

        Returns:
        -------
        H_total : ndarray or list
            The total Hamiltonian (system + control) for the specified
            gate operation and Hamiltonian type.
            - If the gate is "H" or "T", returns a 2x2 matrix.
            - If the gate is "CNOT", returns a 4x4 matrix.
            - For "composite" Hamiltonians, returns a list of Hamiltonians
            representing the sequence.
        """
        if self.gate in ["H", "T"]:
            if self.hamiltonian_type == "field_driven":
                # Driven-Field Hamiltonian
                # System Hamiltonian (static Z field)
                H_sys = (config["hamiltonian"]["OMEGA"] / 2) * \
                    config["pauli_matrices"]["Z_GATE"]
                # Control Hamiltonian (field in XY-plane)
                H_control = amplitude * (
                    np.cos(phase) * config["pauli_matrices"]["X_GATE"]
                    + np.sin(phase) * config["pauli_matrices"]["Y_GATE"]
                )
                return H_sys + H_control

            elif self.hamiltonian_type == "rotational":
                # Rotational Hamiltonian (combination of X and Z fields)
                H_sys = (config["hamiltonian"]["OMEGA"] / np.sqrt(2)) * (
                    config["pauli_matrices"]["X_GATE"] +
                    config["pauli_matrices"]["Z_GATE"]
                )
                H_control = amplitude * (
                    np.cos(phase) * config["pauli_matrices"]["X_GATE"]
                    + np.sin(phase) * config["pauli_matrices"]["Y_GATE"]
                )
                return H_sys + H_control

            elif self.hamiltonian_type == "geometric":
                # Geometric Phase Hamiltonian (pure Z with phase dependence)
                H_sys = (config["hamiltonian"]["OMEGA"] / 2) * \
                    config["pauli_matrices"]["Z_GATE"]
                H_control = (
                    amplitude * config["pauli_matrices"]["Z_GATE"]
                    * np.exp(1j * phase)
                )
                return H_sys + H_control

            elif self.hamiltonian_type == "composite":
                # Composite Pulse Hamiltonian for Hadamard/T gate
                if self.gate == "H":
                    # Hadamard gate decomposition: Z rotation followed
                    # by X rotation
                    H_z = amplitude * config["pauli_matrices"]["Z_GATE"]
                    H_x = amplitude * config["pauli_matrices"]["X_GATE"]
                    return [H_z, H_x]
                elif self.gate == "T":
                    # T gate is a simple Z rotation
                    H_t = amplitude * config["pauli_matrices"]["Z_GATE"]
                    return [H_t]

        elif self.gate == "CNOT":
            if self.hamiltonian_type == "field_driven":
                # Driven-Field CNOT Hamiltonian
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
                )
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
                )
                return H_system + H_control

            elif self.hamiltonian_type == "rotational":
                # Rotational CNOT Hamiltonian (ZX interaction + rotations)
                H_system = config["hamiltonian"]["OMEGA"] * (
                    np.kron(
                        config["pauli_matrices"]["X_GATE"],
                        config["pauli_matrices"]["X_GATE"],
                    )
                    + np.kron(
                        config["pauli_matrices"]["Z_GATE"],
                        config["pauli_matrices"]["Z_GATE"],
                    )
                )
                H_control = amplitude * np.kron(
                    config["pauli_matrices"]["X_GATE"],
                    config["pauli_matrices"]["I_GATE"]
                )
                return H_system + H_control

            elif self.hamiltonian_type == "geometric":
                # Geometric Phase CNOT Hamiltonian
                H_system = (config["hamiltonian"]["OMEGA"] / 2) * np.kron(
                    config["pauli_matrices"]["Z_GATE"],
                    config["pauli_matrices"]["Z_GATE"]
                )
                H_control = (
                    amplitude
                    * np.kron(
                        config["pauli_matrices"]["Z_GATE"],
                        config["pauli_matrices"]["I_GATE"],
                    )
                    * np.exp(1j * phase)
                )
                return H_system + H_control

            elif self.hamiltonian_type == "composite":
                # Composite Pulse Hamiltonian for CNOT
                H_hadamard = amplitude * np.kron(
                    config["pauli_matrices"]["X_GATE"],
                    config["pauli_matrices"]["I_GATE"]
                )  # Hadamard gate for target qubit
                H_cz = amplitude * np.kron(
                    config["pauli_matrices"]["Z_GATE"],
                    config["pauli_matrices"]["Z_GATE"]
                )  # Controlled-Z
                return [H_hadamard, H_cz, H_hadamard]

        else:
            raise ValueError(f"Unsupported gate type: {self.gate}")
