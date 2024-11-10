# def step(self, action):
#     """
#     Takes a step in the environment using the given action.

#     Parameters:
#     ----------
#     action : int
#         The action to be taken by the agent.

#     Returns:
#     -------
#     tuple
#         A tuple containing the next state, reward, and done flag.
#     """
#     self.time_step += 1

#     # Decode action into control pulse parameters
#     amplitude_index = action // (
#         len(self.control_pulse_params["phase"])
#         * len(self.control_pulse_params["duration"])
#     )
#     phase_index = (action // len(self.control_pulse_params["duration"])) % len(
#         self.control_pulse_params["phase"]
#     )
#     duration_index = action % len(self.control_pulse_params["duration"])

#     # Set control pulse parameters
#     amplitude = self.control_pulse_params["amplitude"][amplitude_index]
#     phase = self.control_pulse_params["phase"][phase_index]
#     duration = self.control_pulse_params["duration"][duration_index]

#     # Apply control pulse
#     control_matrix = self._apply_control_pulse(amplitude, phase, duration)
#     next_state = np.dot(control_matrix, self.state)
#     self.state = next_state

#     # Calculate reward
#     reward = -self.infidelity(next_state) if self.time_step == self.max_steps else 0
#     done = self.time_step == self.max_steps

#     # Store state history
#     self.state_history.append(self.state)
#     self.amplitude_history.append(amplitude)
#     self.phase_history.append(phase)
#     self.duration_history.append(duration)

#     return next_state, reward, done, amplitude, phase, duration

# def _apply_control_pulse(self, amplitude, phase, duration):
#     """
#     Applies a control pulse to the quantum gate.

#     Parameters:
#     ----------
#     amplitude (float): The amplitude of the control pulse.
#     phase (float): The phase shift of the control pulse.
#     duration (float): The duration of the control pulse.

#     Returns:
#     -------
#     np.ndarray
#         The resulting unitary matrix after applying the control pulse.
#     """
#     # Calculate the rotation angle
#     theta = amplitude * np.pi * duration

#     # Apply control pulse for single-qubit gates
#     if self.gate in ["H", "T"]:
#         return np.array([
#                     [np.cos(theta / 2), -1j * np.sin(theta / 2) * np.exp(1j * phase)],
#                     [-1j * np.sin(theta / 2) * np.exp(-1j * phase), np.cos(theta / 2)]
#                 ])
#     elif self.gate == "CNOT":
#         # Define the rotation matrix on the target qubit
#         R = np.array([
#                 [np.cos(theta / 2), -1j * np.sin(theta / 2) * np.exp(1j * phase)],
#                 [-1j * np.sin(theta / 2) * np.exp(-1j * phase), np.cos(theta / 2)]
#         ])
#         # Define the identity matrix for the control qubit
#         I = np.eye(2)

#         # Create the full matrix by combining the control and target operations
#         # Note: This assumes the control is the first qubit and target is the second qubit
#         return np.block([
#             [I, np.zeros_like(I)],
#             [np.zeros_like(I), R]
#         ])
#     else:
#         raise ValueError("Unsupported gate type")
