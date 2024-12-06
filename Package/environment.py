import numpy as np
from fidelities import gate_fidelity
from hamiltonians import update_propagator
from hyperparameters import config


class QuantumtEnv:
    """
    A class to represent the environment for quantum gate control
    using reinforcement learning.
    """
    def __init__(
            self,
            gate,
            hamiltonian,
            pulse,
            ):
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
        hamiltonian_type: str
            The type of hamiltonian to be used (e.g., 'field',\
                'rotational', 'geometric', 'composite')
        """
        self.gate = gate
        self.hamiltonian = hamiltonian
        self.pulse = pulse
        self.max_steps = config["hyperparameters"]["MAX_STEPS"]
        self.gate_config = {
            "H": {"num_qubits": 1, "unitary": config['quantum_gates']['HADAMARD_GATE']},
            "T": {"num_qubits": 1, "unitary": config['quantum_gates']['T_GATE']},
            "CNOT": {"num_qubits": 2, "unitary": config['quantum_gates']['CNOT_GATE']}
        }

    def reset(self):
        """
        Resets the environment to the initial state.

        Returns:
        -------
        np.ndarray
            The initial quantum state.
        """
        self.time_step = 0
        self.last_action = -1
        self.U_accumulated = self._get_initial_propagator()
        self.state = self._get_initial_state().copy()
        # Initialize the episode dictionary
        self.episode_data = {
            "discounted_reward": [],
            "fidelity": [],
            "state": [self.state],
            "U_accumulated": [self.U_accumulated],
            "control_pulse_params": {
                key: [] for key in self.pulse.control_pulse_params
            },
        }

        return self._compute_state(
            self.U_accumulated,
            self.last_action,
            self.time_step,
            self.max_steps
        )

    def step(self, action):
        raise NotImplementedError("Subclasses must implement the step method.")

    def _get_initial_propagator(self):
        raise NotImplementedError(
            "Subclasses must implement `_get_initial_propagator`."
        )

    def _get_initial_state(self):
        raise NotImplementedError("Subclasses must implement `_get_initial_state`.")

    def _compute_state(self, propagator, last_action, time_step, max_steps):
        """
        Compute the state given the propagator, last action, current time step and maximum number of steps.

        Parameters
        ----------
        propagator : 2D complex array
            The propagator matrix at the current time step.
        last_action : int
            The index of the last action taken.
        time_step : int
            The current time step.
        max_steps : int
            The maximum number of steps.

        Returns
        -------
        state : 1D complex array
            The state vector, which is a concatenation of the real and imaginary parts of the propagator, the one-hot encoded last action and the normalized time step.
        """

        real_parts = np.real(propagator).flatten()
        imag_parts = np.imag(propagator).flatten()

        # Encode last action
        # Add last action directly
        action_encoded = np.array([last_action / (self.action_size - 1)])

        normalized_time = np.array([time_step / max_steps])

        state = np.concatenate([
            real_parts,
            imag_parts,
            action_encoded,
            normalized_time
        ])

        return state

    def _log_gate_infidelity(self, final_unitary, target_unitary):
        """
        Calculates the infidelity between the final state and
        the theoretical state.

        Parameters:
        ----------
        final_state (np.ndarray): The final quantum state after applying
        the control pulse.

        Returns:
        -------
        Tuple[float, float]
            A tuple of log-infidelity and fidelity.
        """
        try:
            fidelity = gate_fidelity(final_unitary, target_unitary)
            if fidelity >= 1.0:
                return np.inf, fidelity
            return -np.log10(1 - fidelity), fidelity
        except Exception as e:
            print(f"Error calculating infidelity: {e}")
            return None, None

    def _get_reward(
            self,
            fidelity,
            log_infidelity,
            fidelity_penalty_drop,
            fidelity_average,
            fidelity_variance,
            time_step,
            max_steps
            ):
        """
        Calculate the reward for the current time step based on fidelity.

        Parameters:
            fidelity (float): The gate fidelity (F).
            log_infidelity (float): The logarithmic infidelity (-log(1-F)).
            fidelity_penalty_drop (float): The difference between the
            fidelity at the previous time step and the current fidelity.
            fidelity_average (float): The average fidelity over the episode.
            fidelity_variance (float): The variance of the fidelity over the
            episode.
            time_step (int): The current time step.
            max_steps (int): The maximum number of steps in the episode.

        Returns:
            float: The reward value.
            bool: Whether the episode should end early.
        """

        # Logarithmic reward based on infidelity
        if fidelity < 1.0:
            reward = log_infidelity
        else:
            reward = 0  # Avoid log(0) issues

        # Clipping reward
        if 0.95 <= fidelity < 0.99:
            reward = 1

        # Bonus reward
        if fidelity >= 0.99:
            reward += 10
            done = True  # End the episode early
        else:
            done = time_step >= max_steps  # Check max steps

        reward += (
            - config["hyperparameters"]["FIDELITY_DROP_PENALTY_WEIGHT"]
            * fidelity_penalty_drop
            + config["hyperparameters"]["FIDELITY_AVERAGE_WEIGHT"] * fidelity_average
            - config["hyperparameters"]["FIDELITY_VARIANCE_WEIGHT"] * fidelity_variance
        )

        # Apply discount to the reward
        discounted_reward = reward * (config['hyperparameters']['GAMMA'] ** time_step)

        return discounted_reward, done


class OneQubitEnv(QuantumtEnv):
    """
    A class representing a one-qubit quantum environment."""

    def __init__(self, gate, hamiltonian, pulse):
        """
        Initializes the single-qubit environment.
        """
        super().__init__(gate, hamiltonian, pulse)

        self.num_qubits = self.gate_config[self.gate]["num_qubits"]
        self.target_unitary = self.gate_config[self.gate]["unitary"]
        self.input_features = 2 ** (2 * self.num_qubits + 1) + 1 + 1
        self.action_size = self.pulse.total_actions
        self.initial_propagator = np.eye(2**self.num_qubits, dtype=np.complex128)
        self.initial_state = self._compute_state(
            self.initial_propagator, -1, 0, self.max_steps
        )

    def reset(self):
        """
        Resets the single-qubit environment, initializing the propagator and parameters.
        """
        super().reset()  # Call the base class reset logic
        return self.state

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

        # Generate control pulse and construct hamiltonian
        omega, delta = self.pulse.generate_control_pulse(action)
        hamiltonian = self.hamiltonian.build_one_qubit_hamiltonian(
            omega,
            delta
        )

        # Update accumulated propagator
        self.U_accumulated = update_propagator(
            self.U_accumulated,
            hamiltonian,
            self.time_step,
            self.max_steps
        )

        # Update last action
        self.last_action = action

        # Compute the next state
        next_state = self._compute_state(
            self.U_accumulated,
            self.last_action,
            self.time_step,
            self.max_steps
        )

        # Update time step
        self.time_step += 1

        # Compute log_infidelity, fidelity, fidelity_penalty, fidelity average and fidelity variance
        log_infidelity, fidelity = self._log_gate_infidelity(
            final_unitary=self.U_accumulated,
            target_unitary=self.target_unitary
        )

        # Log episode data
        self.episode_data["state"].append(next_state)
        self.episode_data["U_accumulated"].append(self.U_accumulated)
        self.episode_data["control_pulse_params"]["omega"].append(omega)
        self.episode_data["control_pulse_params"]["delta"].append(delta)
        self.episode_data["fidelity"].append(fidelity)

        fidelity_penalty_drop = (
            self.episode_data["fidelity"][-2] - self.episode_data["fidelity"][-1]
            if len(self.episode_data["fidelity"]) > 1
            else 0
        )
        fidelity_average = (
            np.mean(self.episode_data["fidelity"])
            if len(self.episode_data["fidelity"]) > 1
            else 0
        )
        fidelity_variance = (
            np.var(self.episode_data["fidelity"], ddof=1)
            if len(self.episode_data["fidelity"]) > 1
            else 0
        )

        discounted_reward, done = self._get_reward(
            fidelity,
            log_infidelity,
            fidelity_penalty_drop,
            fidelity_average,
            fidelity_variance,
            self.time_step,
            self.max_steps,
        )

        self.episode_data["discounted_reward"].append(discounted_reward)

        return (
            done,
            next_state,
            discounted_reward,
            fidelity,
            log_infidelity,
        )

    def _get_initial_propagator(self):
        """
        Returns the initial propagator for a single-qubit system.
        """
        return np.eye(2, dtype=np.complex128)

    def _get_initial_state(self):
        """
        Returns the initial state for a single-qubit system.
        """
        return self._compute_state(
            self.U_accumulated, -1, self.time_step, self.max_steps
        )


class TwoQubitEnv(QuantumtEnv):
    """
    A class representing a two-qubit quantum environment.
    """
    def __init__(self, gate, hamiltonian, pulse):
        """
        Initializes the single-qubit environment.
        """
        super().__init__(
            gate, hamiltonian, pulse)
        self.num_qubits = self.gate_config[self.gate]["num_qubits"]
        self.target_unitary = self.gate_config[self.gate]["unitary"]
        self.input_features = 2 ** (2 * self.num_qubits + 1) + 1 + 1
        self.action_size = self.pulse.total_actions
        self.initial_propagator = np.eye(4, dtype=np.complex128)
        self.initial_state = self._compute_state(
            self.initial_propagator, -1, 0, self.max_steps
        )

    def reset(self):
        """
        Resets the two-qubit environment, initializing the propagator and parameters.
        """
        super().reset()  # Call the base class reset logic
        return self.state

    def step(self, action):

        omega1, delta1, omega2, delta2, coupling_strength = self.pulse.generate_control_pulse(action)
        hamiltonian = self.hamiltonian.build_two_qubit_hamiltonian(
            omega1, delta1, omega2, delta2, coupling_strength
        )

        self.U_accumulated = update_propagator(
            self.U_accumulated,
            hamiltonian,
            self.time_step,
            self.max_steps,
        )

        self.last_action = action
        next_state = self._compute_state(
            self.U_accumulated,
            self.last_action,
            self.time_step,
            self.max_steps
        )

        log_infidelity, fidelity = self._log_gate_infidelity(
            final_unitary=self.U_accumulated,
            target_unitary=self.target_unitary
        )

        self.time_step += 1

        self.episode_data["state"].append(next_state)
        self.episode_data["U_accumulated"].append(self.U_accumulated)
        self.episode_data["control_pulse_params"]["omega1"].append(omega1)
        self.episode_data["control_pulse_params"]["delta1"].append(delta1)
        self.episode_data["control_pulse_params"]["omega2"].append(omega2)
        self.episode_data["control_pulse_params"]["delta2"].append(delta2)
        self.episode_data["control_pulse_params"]["coupling_strength"].append(
            coupling_strength
        )
        self.episode_data["fidelity"].append(fidelity)

        # Update the episode data dictionary
        self.episode_data["state"].append(next_state)
        self.episode_data["U_accumulated"].append(self.U_accumulated)
        self.episode_data["control_pulse_params"]["omega1"].append(omega1)
        self.episode_data["control_pulse_params"]["delta1"].append(delta1)
        self.episode_data["control_pulse_params"]["omega2"].append(omega2)
        self.episode_data["control_pulse_params"]["delta2"].append(delta2)
        self.episode_data["control_pulse_params"]["coupling_strength"].append(coupling_strength)
        self.episode_data["fidelity"].append(fidelity)

        fidelity_penalty_drop = (
            self.episode_data["fidelity"][-2] - self.episode_data["fidelity"][-1]
            if len(self.episode_data["fidelity"]) > 1 else 0
        )
        fidelity_average = np.mean(self.episode_data["fidelity"]) if len(self.episode_data["fidelity"]) > 1 else 0
        fidelity_variance = np.var(self.episode_data["fidelity"], ddof=1) if len(self.episode_data["fidelity"]) > 1 else 0

        discounted_reward, done = self._get_reward(
            fidelity,
            log_infidelity,
            fidelity_penalty_drop,
            fidelity_average,
            fidelity_variance,
            self.time_step,
            self.max_steps,
        )

        self.episode_data["discounted_reward"].append(discounted_reward)

        return (
            done,
            next_state,
            discounted_reward,
            fidelity,
            log_infidelity,
        )

    def _get_initial_propagator(self):
        """
        Returns the initial propagator for a two-qubit system.
        """
        return np.eye(4, dtype=np.complex128)

    def _get_initial_state(self):
        """
        Returns the initial state for a two-qubit system.
        """
        return self._compute_state(
            self.U_accumulated, -1, self.time_step, self.max_steps
        )
