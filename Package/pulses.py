import numpy as np


class PulseGenerator:
    def __init__(self, control_pulse_type, gate):
        self.control_pulse_type = control_pulse_type
        self.gate = gate
        self.total_actions = 0
        if self.control_pulse_type == "Discrete":
            if self.gate in ["H", "T"]:
                self.number_points = 10
                self.control_pulse_params = {
                    "omega": np.linspace(0, 2 * np.pi, self.number_points),
                    "delta": np.linspace(-2 * np.pi, 2 * np.pi, self.number_points),
                }
                self.bounds = {
                    "omega": (0, 2 * np.pi, self.number_points),
                    "delta": (-2 * np.pi, 2 * np.pi, self.number_points),
                }
                self.total_actions = self.number_points ** 2
            else:
                self.number_points = 10
                self.control_pulse_params = {
                    "omega1": np.linspace(0, 2 * np.pi, self.number_points),
                    "omega2": np.linspace(0, 2 * np.pi, self.number_points),
                    "delta1": np.linspace(-2 * np.pi, 2 * np.pi, self.number_points),
                    "delta2": np.linspace(-2 * np.pi, 2 * np.pi, self.number_points),
                    "coupling_strength": np.linspace(
                        0, 10, self.number_points
                    ),
                }
                self.bounds = {
                    "omega1": (0, 2 * np.pi, self.number_points),
                    "omega2": (0, 2 * np.pi, self.number_points),
                    "delta1": (-2 * np.pi, 2 * np.pi, self.number_points),
                    "delta2": (-2 * np.pi, 2 * np.pi, self.number_points),
                    "coupling_strength": (0, 10, self.number_points),
                }
                self.total_actions = self.number_points ** 5

    def generate_control_pulse(self, action):
        """
        Generates a control pulse based on the given action
        and control pulse type.

        Parameters:
        - action (int): Action to be interpreted for generating the pulse.

        Returns:
        - omega (float): Rabi frequency.
        - delta (float): Detuning.
        - J (float): Coupling strength.
        """
        if self.control_pulse_type == "Discrete":
            if self.gate in ["H", "T"]:
                num_deltas = len(self.control_pulse_params["delta"])
 
                if not (0 <= action < self.total_actions):
                    raise ValueError(
                        f"Action {action} out of bounds. Valid range: 0 to {self.total_actions - 1}."
                    )

                omega_index = action // num_deltas
                delta_index = action % num_deltas

                omega = self.control_pulse_params["omega"][omega_index]
                delta = self.control_pulse_params["delta"][delta_index]

                # self._validate_control_pulse(omega, delta)
                return omega, delta

            elif self.gate in ["CNOT"]:
                # Generate for Omega1, Omega2, Delta1, Delta2, and J
                num_deltas_1 = len(self.control_pulse_params["delta1"])
                num_omegas_1 = len(self.control_pulse_params["omega1"])
                num_deltas_2 = len(self.control_pulse_params["delta2"])
                num_omegas_2 = len(self.control_pulse_params["omega2"])
                num_J = len(self.control_pulse_params["coupling_strength"])

                if not (0 <= action < self.total_actions):
                    raise ValueError(
                        f"Action {action} out of bounds. Valid range: 0 to {self.total_actions - 1}."
                    )

                # Cumulative multipliers for decoding
                multiplier_omega2 = num_deltas_2 * num_omegas_1 * num_deltas_1 * num_J
                multiplier_delta2 = num_omegas_1 * num_deltas_1 * num_J
                multiplier_omega1 = num_deltas_1 * num_J
                multiplier_delta1 = num_J

                # Decode indices
                omega1_index = (action // multiplier_omega1) % num_omegas_1
                delta1_index = (action // multiplier_delta1) % num_deltas_1

                omega2_index = (action // multiplier_omega2) % num_omegas_2
                delta2_index = (action // multiplier_delta2) % num_deltas_2

                coupling_strength_index = action % num_J

                # Extract parameters
                omega1 = self.control_pulse_params["omega1"][omega1_index]
                delta1 = self.control_pulse_params["delta1"][delta1_index]

                omega2 = self.control_pulse_params["omega2"][omega2_index]
                delta2 = self.control_pulse_params["delta2"][delta2_index]

                coupling_strength = self.control_pulse_params["coupling_strength"][coupling_strength_index]

                # self._validate_control_pulse(omega1, delta1)
                # self._validate_control_pulse(omega2, delta2, coupling_strength)

                # Return parameters
                return omega1, delta1, omega2, delta2, coupling_strength
