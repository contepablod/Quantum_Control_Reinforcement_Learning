import numpy as np


class PulseGenerator:
    def __init__(self, control_pulse_type, gate, hamiltonian_type, agent_type):
        self.control_pulse_type = control_pulse_type
        self.gate = gate
        self.action_size = None
        self.hamiltonian_type = hamiltonian_type
        self.agent_type = agent_type
        self._initialize_params()

    def _initialize_params(self):
        if self.control_pulse_type == "Discrete":
            self._initialize_discrete_params()
        elif self.control_pulse_type == "Continuous":
            self._initialize_continuous_params()

    def _initialize_discrete_params(self):
        self.number_actions = 3
        if self.gate in ["H", "T"]:
            if self.hamiltonian_type == "Field":
                self.control_pulse_params = {
                    "omega": np.linspace(-2, 2, self.number_actions),
                    "delta": np.linspace(-2, 2, self.number_actions),
                }
                if self.agent_type == "PPO" or self.agent_type == "GP":
                    self.action_size = self.number_actions
                else:
                    self.action_size = self.number_actions ** len(
                        self.control_pulse_params.keys()
                    )

            elif self.hamiltonian_type == "Geometric":
                self.control_pulse_params = {
                    "omega": np.linspace(0, 4, self.number_actions),
                    "delta": np.linspace(-4, 4, self.number_actions),
                    "phase": np.linspace(-np.pi, np.pi, self.number_actions),
                }
                self.action_size = self.number_actions
        else:
            if self.hamiltonian_type == "Field":
                self.control_pulse_params = {
                    "omega1": np.linspace(-2, 2, self.number_actions),
                    "delta1": np.linspace(-2, 2, self.number_actions),
                    "omega2": np.linspace(-2, 2, self.number_actions),
                    "delta2": np.linspace(-2, 2, self.number_actions),
                    # "coupling_strength_zx": np.linspace(-2, 2, self.number_actions),
                    "coupling_strength_zx": np.array([-2.0, -1.0, 1.0, 2.0]),
                    # "coupling_strength_zz": np.array([-2.0, -1.0, 1.0, 2.0]),
                }
                if self.agent_type == "PPO" or self.agent_type == "GP":
                    self.action_size = self.number_actions
                else:
                    self.action_size = self.action_size = self.number_actions ** (
                        len(self.control_pulse_params.keys()) - 1
                    ) * (self.number_actions - 1)
                    # Get the action for 5x5x5x5x4

            elif self.hamiltonian_type == "Geometric":
                self.control_pulse_params = {
                    "omega1": np.linspace(-1, 1, self.number_actions),
                    "delta1": np.linspace(-1, 1, self.number_actions),
                    "phase1": np.linspace(-np.pi, np.pi, self.number_actions),
                    "omega2": np.linspace(-1, 1, self.number_actions),
                    "delta2": np.linspace(-1, 1, self.number_actions),
                    "phase2": np.linspace(-np.pi, np.pi, self.number_actions),
                    "coupling_strength_zx": np.array([-4.0, -2.0, 2.0, 4.0]),
                }
                self.action_size = self.number_actions ** (
                    len(self.control_pulse_params.keys()) - 1
                ) * (self.number_actions - 1)

    def _initialize_continuous_params(self):
        if self.gate in ["H", "T"]:
            if self.hamiltonian_type == "Field":
                self.control_pulse_params = {
                    "omega": (-4, 4),
                    "delta": (-4, 4),
                }
                self.action_size = len(self.control_pulse_params.keys())

            elif self.hamiltonian_type == "Geometric":
                self.control_pulse_params = {
                    "omega": (-4, 4 + 1e-8),
                    "delta": (-4, 4 + 1e-8),
                    "phase": (-np.pi, np.pi + 1e-8),
                }
                self.action_size = len(self.control_pulse_params.keys())
        else:
            if self.hamiltonian_type == "Field":
                self.control_pulse_params = {
                    "omega1": (-4, 4),
                    "delta1": (-4, 4),
                    "omega2": (-4, 4),
                    "delta2": (-4, 4),
                    "coupling_strength_zx": (-4, 4),
                }
                self.action_size = len(self.control_pulse_params.keys())

            # elif self.hamiltonian_type == 'Geometric':
            #     self.control_pulse_params = {
            #         "omega1": (-1, 1),
            #         "delta1": (-1, 1),
            #         "phase1": (-np.pi, np.pi),
            #         "omega2": (-1, 1),
            #         "delta2": (-1, 1),
            #         "phase2": (-np.pi, np.pi),
            #     }
            #     self.action_size = len(self.control_pulse_params.keys())

        self.control_pulse_params = {key: [] for key in self.control_pulse_params.keys()}

    def generate_control_pulse(self, action=None):
        if self.control_pulse_type == "Discrete":
            pulse_values = self._generate_discrete_pulse(action)
        elif self.control_pulse_type == "Continuous":
            pulse_values = self._generate_continuous_pulse(action)
        return (
            {
                key: value
                for key, value in zip(self.control_pulse_params.keys(), pulse_values)
            }
            if isinstance(pulse_values, (list, np.ndarray, tuple))
            else (
                {list(self.control_pulse_params.keys())[0]: pulse_values.item()}
                if isinstance(pulse_values, (np.float64, float, int))
                else pulse_values
            )
        )

    def _generate_discrete_pulse(self, action):
        if self.hamiltonian_type == "Field":
            return self._generate_field_discrete_pulse(action)
        elif self.hamiltonian_type == "Geometric":
            return self._generate_geometric_discrete_pulse(action)

    def _generate_continuous_pulse(self, action):
        if self.hamiltonian_type == "Field":
            return self._generate_field_continuous_pulse(action)
        elif self.hamiltonian_type == "Geometric":
            return self._generate_geometric_continuous_pulse(action)

    def _generate_field_discrete_pulse(self, action):
        if self.gate in ["H", "T"]:
            if self.agent_type == "PPO" or self.agent_type == "GP":
                omega_index = action[0]
                delta_index = action[1]
            else:
                omega_index = action // self.number_actions
                delta_index = action % self.number_actions
            omega = self.control_pulse_params["omega"][omega_index]
            delta = self.control_pulse_params["delta"][delta_index]
            return omega, delta

        else:
            # For the two-qubit system, we assume that control_pulse_params is defined as follows:
            # "omega1", "delta1", "omega2", "delta2", "coupling_strength_zx",
            if self.agent_type == "PPO" or self.agent_type == "GP":
                omega1_index = action[0]
                delta1_index = action[1]
                omega2_index = action[2]
                delta2_index = action[3]
                coupling_zx_index = action[4]
            else:
                num_omega1 = len(self.control_pulse_params["omega1"])
                num_delta1 = len(self.control_pulse_params["delta1"])
                num_omega2 = len(self.control_pulse_params["omega2"])
                num_delta2 = len(self.control_pulse_params["delta2"])
                num_coupling_zx = len(self.control_pulse_params["coupling_strength_zx"])

                # Decompose the flattened action index into individual parameter indices.
                # The ordering used here is: omega1, delta1, omega2, delta2, coupling_strength_zx, coupling_strength_zz
                index = action

                coupling_zx_index = index % num_coupling_zx
                index //= num_coupling_zx

                delta2_index = index % num_delta2
                index //= num_delta2

                omega2_index = index % num_omega2
                index //= num_omega2

                delta1_index = index % num_delta1
                index //= num_delta1

                omega1_index = index % num_omega1

            # Extract the parameters from control_pulse_params using the indices.
            omega1 = self.control_pulse_params["omega1"][omega1_index]
            delta1 = self.control_pulse_params["delta1"][delta1_index]
            omega2 = self.control_pulse_params["omega2"][omega2_index]
            delta2 = self.control_pulse_params["delta2"][delta2_index]
            coupling_strength_zx = self.control_pulse_params["coupling_strength_zx"][
                coupling_zx_index
            ]

            return omega1, delta1, omega2, delta2, coupling_strength_zx

    def _generate_geometric_discrete_pulse(self, action):
        if self.gate in ["H", "T"]:
            num_phase = len(self.control_pulse_params["phase"])
            num_delta = len(self.control_pulse_params["delta"])
            total_delta_phase = num_delta * num_phase
            omega_index = action // total_delta_phase
            remainder = action % total_delta_phase
            delta_index = remainder // num_phase
            phase_index = remainder % num_phase
            omega = self.control_pulse_params["omega"][omega_index]
            delta = self.control_pulse_params["delta"][delta_index]
            phase = self.control_pulse_params["phase"][phase_index]

            return omega, delta, phase

        # elif self.gate == "CNOT":
        #     num_deltas_1 = len(self.control_pulse_params["delta1"])
        #     num_omegas_1 = len(self.control_pulse_params["omega1"])
        #     num_deltas_2 = len(self.control_pulse_params["delta2"])
        #     num_omegas_2 = len(self.control_pulse_params["omega2"])
        #     num_J_xz = len(self.control_pulse_params["coupling_strength"])
        #     num_J_xx = len(self.control_pulse_params["coupling_strength"])

        #     multiplier_omega2 = num_deltas_2 * num_omegas_1 * num_deltas_1 * num_J
        #     multiplier_delta2 = num_omegas_1 * num_deltas_1 * num_J
        #     multiplier_omega1 = num_deltas_1 * num_J
        #     multiplier_delta1 = num_J

        #     omega1_index = (action // multiplier_omega1) % num_omegas_1
        #     delta1_index = (action // multiplier_delta1) % num_deltas_1
        #     omega2_index = (action // multiplier_omega2) % num_omegas_2
        #     delta2_index = (action // multiplier_delta2) % num_deltas_2
        #     coupling_strength_index = action % num_J

        #     omega1 = self.control_pulse_params["omega1"][omega1_index]
        #     delta1 = self.control_pulse_params["delta1"][delta1_index]
        #     omega2 = self.control_pulse_params["omega2"][omega2_index]
        #     delta2 = self.control_pulse_params["delta2"][delta2_index]
        #     coupling_strength = self.control_pulse_params["coupling_strength"][coupling_strength_index]

        #     return omega1, delta1, omega2, delta2, coupling_strength

    def _generate_field_continuous_pulse(self, action):
        if self.gate in ["H", "T"]:
            pulse_params = {"omega": action[0], "delta": action[1]}
        else:
            pulse_params = {
                "omega1": action[0],
                "delta1": action[1],
                "omega2": action[2],
                "delta2": action[3],
                "coupling_strength_zx": action[4],
            }
        return tuple(pulse_params[key] for key in self.control_pulse_params.keys())

    def _generate_geometric_continuous_pulse(self, action):
        if self.gate in ["H", "T"]:
            pulse_params = {"omega": action[0], "delta": action[1], "phase": action[2]}
        return tuple(pulse_params[key] for key in self.control_pulse_params.keys())
