import numpy as np
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
            self.hidden_features,
            self.epsilon_decay,
        ) = self.set_quantum_info()
        self.reset()

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
            initial_states = [
                np.array([1, 0], dtype=np.complex128),
                np.array([0, 1], dtype=np.complex128),
            ]
            initial_state = initial_states[np.random.choice(len(initial_states))]
            input_features = 2 ** (
                num_qubits + 1
            )  # Number of states in the input space
            hidden_features = 512
            epsilon_decay = 0.95
            unitary = HADAMARD if self.gate == "H" else T_GATE
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
            hidden_features = 512
            epsilon_decay = 0.9995
            unitary = CNOT
            target_state = np.dot(initial_state, unitary)

        return (
            initial_state,
            target_state,
            state_size,
            action_size,
            input_features,
            hidden_features,
            epsilon_decay,
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
        self.max_steps = MAX_STEPS
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
            H_sys = (omega / 2) * Z  # System Hamiltonian (static)
            H_control = amplitude * (
                np.cos(phase) * X + np.sin(phase) * Y
            )  # Control Hamiltonian (depends on amplitude and phase)
            return H_sys + H_control  # Total Hamiltonian
        elif self.gate == "CNOT":
            H_system = (
                (omega / 2) * np.kron(Z, I)
                + (omega / 2) * np.kron(I, Z)
                + J * (np.kron(X, X) + np.kron(Y, Y))
            )  # System Hamiltonian (static)
            H_control = amplitude * (
                np.kron(X, I) * np.cos(phase) + np.kron(Y, I) * np.sin(phase)
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
        fidelity = np.abs(np.dot(np.conjugate(self.target_state), final_state)) ** 2
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
