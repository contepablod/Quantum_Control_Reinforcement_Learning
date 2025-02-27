import numpy as np
from fidelities import log_gate_infidelity
from hyperparameters import config
from scipy.linalg import expm


class QuantumtEnv:
    def __init__(
            self,
            gate,
            hamiltonian,
            pulse,
            device
            ):
        self.gate = gate
        self.hamiltonian = hamiltonian
        self.pulse = pulse
        self.device = device
        self.control_pulse_type = self.pulse.control_pulse_type
        self.action_size = self.pulse.action_size
        self.max_steps = config["hyperparameters"]["general"]["MAX_STEPS"]
        self.gamma = config["hyperparameters"]["general"]["GAMMA"]
        self.num_qubits = config["quantum_gates"][self.gate]["num_qubits"]
        self.target_unitary = config["quantum_gates"][self.gate]["unitary"]

        self.fidelity_threshold = config["hyperparameters"]["train"][
            "FIDELITY_THRESHOLD"
            ]
        self.input_features = 2 ** (2 * self.num_qubits + 1)
        self.initial_propagator = np.eye(
            2**self.num_qubits,
            dtype=np.complex128
        )
        self.initial_state = self._compute_state(
            self.initial_propagator
        )
        self.total_time = config["hyperparameters"]["general"]["TOTAL_TIME"]
        self.time_delta = self.total_time / self.max_steps
        self.reset()

    def reset(self):
        self.time_step = 0
        self.U_accumulated = self.initial_propagator
        self.state = self.initial_state
        self.episode_data = {
            "step_reward": [],
            "fidelity": [],
            "avg_fidelity": [],
            "log_infidelity": [],
            "state": [],
            "U_accumulated": [],
            "control_pulse_params": [],
        }
        return self.state

    def step(self, action):
        raise NotImplementedError("Subclasses must implement the step method.")

    def _compute_state(self, propagator):
        # Flatten propagator
        real_parts = np.real(propagator).ravel()
        imag_parts = np.imag(propagator).ravel()
        # Concatenate elements
        state = np.concatenate([real_parts, imag_parts])
        return state

    def _get_reward(
            self,
            log_infidelity,
            fidelity
            ):
        # Compute reward penalizing longer episodes
        reward = log_infidelity  # * (1 - self.time_step / self.max_steps)
        done = False
        if (
            self.time_step == self.max_steps or fidelity >= self.fidelity_threshold
        ):
            # Reward for reaching the target fidelity or timeout
            done = True
            if fidelity >= self.fidelity_threshold:
                reward *= 5

        return reward, done

    def _store_step_metrics(
            self,
            pulse_params,
            U_accumulated,
            next_state,
            fidelity_gate,
            log_infidelity,
            avg_gate_fidelity,
            step_reward
            ):

        self.episode_data["control_pulse_params"].append(pulse_params)
        self.episode_data["U_accumulated"].append(U_accumulated)
        self.episode_data["state"].append(next_state)
        self.episode_data["fidelity"].append(fidelity_gate)
        self.episode_data["avg_fidelity"].append(avg_gate_fidelity)
        self.episode_data["log_infidelity"].append(log_infidelity)
        self.episode_data["step_reward"].append(step_reward)


class GateEnv(QuantumtEnv):
    def __init__(self, gate, hamiltonian, pulse, device):
        super().__init__(gate, hamiltonian, pulse, device)

    def reset(self):
        super().reset()
        return self.state

    def step(self, action):
        # Update time
        self.time_step += 1

        # Generate control pulse
        self.pulse_params = self.pulse.generate_control_pulse(action)

        # Setup hamiltonian
        if self.gate in ["H", "T"]:
            hamiltonian = self.hamiltonian.build_one_qubit_hamiltonian(self.pulse_params)
        elif self.gate == 'CNOT':
            hamiltonian = self.hamiltonian.build_two_qubit_hamiltonian(self.pulse_params)

        # Compute the step propagator
        U_step = expm(-1j * hamiltonian * self.time_delta)

        # Update the accumulated propagator
        self.U_accumulated = self.U_accumulated @ U_step

        # Compute the next state
        self.next_state = self._compute_state(self.U_accumulated)

        # Compute log_infidelity, fidelity, avg_fidelity
        (
            self.log_infidelity,
            self.fidelity_gate,
            self.avg_gate_fidelity
        ) = log_gate_infidelity(
            final_unitary=self.U_accumulated,
            target_unitary=self.target_unitary
        )
        # Compute reward
        (
            self.step_reward,
            self.done
        ) = self._get_reward(
            self.log_infidelity,
            self.fidelity_gate
        )

        # Log episode data
        self._store_step_metrics(
            self.pulse_params,
            self.U_accumulated,
            self.next_state,
            self.fidelity_gate,
            self.log_infidelity,
            self.avg_gate_fidelity,
            self.step_reward
        )

        return self.done, self.next_state, self.step_reward
