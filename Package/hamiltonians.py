import numpy as np
from hyperparameters import config
from scipy.linalg import expm, polar
from numpy.linalg import qr


class Hamiltonian:
    def __init__(self, hamiltonian_type, gate):
        self.hamiltonian_type = hamiltonian_type
        self.gate = gate
        self.X = np.array(config["pauli_matrices"]["X_GATE"],
                          dtype=np.complex128)
        self.Y = np.array(config["pauli_matrices"]["Y_GATE"],
                          dtype=np.complex128)
        self.Z = np.array(config["pauli_matrices"]["Z_GATE"],
                          dtype=np.complex128)
        self.I = np.eye(2, dtype=np.complex128)  # Identity matrix

    def build_one_qubit_hamiltonian(self, omega, delta):
        """
        Constructs the Hamiltonian for the quantum gate operation based
        on the gate type and Hamiltonian type.

        Parameters:
        ----------
        omega: float
            The Rabi frequency of the control pulse.
        delta : float
            The detuning of the control pulse.
        J : float, optional
            The coupling constant for the CNOT gate. Defaults to 0.
        hamiltonian_type : str
            The type of Hamiltonian to use ("field", "rotational",
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
        if self.hamiltonian_type == "Field":
            return self._field_hamiltonian(omega, delta)
        elif self.hamiltonian_type == "Rotational":
            return self._rotational_hamiltonian(omega)
        elif self.hamiltonian_type == "Geometric":
            return self._geometric_hamiltonian(omega, delta)
        elif self.hamiltonian_type == "Composite":
            return self._composite_hamiltonian(omega, delta)

    def build_two_qubit_hamiltonian(self, omega1, delta1, omega2, delta2, coupling_strength):
        if self.hamiltonian_type == "Field":
            return self._field_hamiltonian_cnot(omega1, delta1, omega2, delta2, coupling_strength)
        # elif self.hamiltonian_type == "rotational":
        #     return self._rotational_hamiltonian_cnot(omega1, coupling_strength)
        # elif self.hamiltonian_type == "geometric":
        #     return self._geometric_hamiltonian_cnot(omega1, delta1, coupling_strength)
        # elif self.hamiltonian_type == "composite":
        #     return self._composite_hamiltonian_cnot(omega1, delta1, omega2, coupling_strength)

    # ========================
    # Single-Qubit Hamiltonians
    # ========================

    def _field_hamiltonian(self, omega, delta):
        """Field Hamiltonian for single-qubit gates."""
        H_control = omega * self.X
        H_system = delta * self.Z
        return H_system + H_control

    def _rotational_hamiltonian(self, omega):
        """Rotational Hamiltonian for single-qubit gates."""
        return omega * self.X

    def _geometric_hamiltonian(self, omega, delta):
        """Geometric Hamiltonian for single-qubit gates."""
        return omega * self.X + delta * self.Z

    def _composite_hamiltonian(self, omega, delta, phase):
        """Composite Hamiltonian for single-qubit gates."""
        H1 = self._field_hamiltonian(omega, delta)
        H2 = self._geometric_hamiltonian(omega, delta)
        return [H1, H2]

    # ========================
    # Two-Qubit Hamiltonians
    # ========================

    def _field_hamiltonian_cnot(
        self, omega1, delta1, omega2, delta2, coupling_strength
    ):
        """
        Field Hamiltonian for CNOT gate with detuning.

        Parameters:
        ----------
        omega1 : float
            Rabi frequency for the control qubit.
        delta1 : float
            Detuning for the control qubit.
        omega2 : float
            Rabi frequency for the target qubit.
        delta2 : float
            Detuning for the target qubit.
        coupling_strength : float
            Coupling strength between the qubits.

        Returns:
        -------
        H_total : ndarray
            The field Hamiltonian for the CNOT gate.
        """
        # Control qubit Hamiltonian (control field + detuning)
        H_control1 = omega1 * (np.kron(self.X, self.I) + np.kron(self.Y, self.I))
        H_detuning1 = delta1 * np.kron(
            self.Z, self.I
        )  # Detuning term for control qubit

        # Target qubit Hamiltonian (control field + detuning)
        H_control2 = omega2 * (np.kron(self.I, self.X) + np.kron(self.I, self.Y))
        H_detuning2 = delta2 * np.kron(self.I, self.Z)  # Detuning term for target qubit

        # Interaction Hamiltonian
        H_interaction = coupling_strength * np.kron(self.Z, self.Z)

        # Total Hamiltonian
        H_total = H_control1 + H_detuning1 + H_control2 + H_detuning2 + H_interaction

        return H_total

    # def _rotational_hamiltonian_cnot(self, omega, phase, coupling_strength):
    #     """Rotational Hamiltonian for CNOT gate."""
    #     return omega * np.kron(self.X, self.I) + coupling_strength * np.kron(
    #         self.Z, self.Z
    #     )

    # def _geometric_hamiltonian_cnot(self, omega, delta, coupling_strength):
    #     """Geometric Hamiltonian for CNOT gate."""
    #     H_control = omega * np.kron(self.X, self.I)
    #     H_interaction = coupling_strength * np.kron(self.Z, self.Z)
    #     return H_control + H_interaction

    # def _composite_hamiltonian_cnot(self, omega, delta, phase, coupling_strength):
    #     """Composite Hamiltonian for CNOT gate."""
    #     H1 = self._field_hamiltonian_cnot(
    #         omega1, delta1, omega2, delta2, coupling_strength
    #     )
    #     H2 = self._geometric_hamiltonian_cnot(omega1, delta1, coupling_strength)
    #     return [H1, H2]


def update_propagator(
        U_accumulated,
        hamiltonian,
        time_step,
        max_steps,
        polar_project=False,
        qr_project=False):
    """
    Updates the accumulated propagator with a given Hamiltonian.

    Parameters:
        U_accumulated (np.ndarray): The current accumulated propagator
        (unitary matrix).
        hamiltonian (np.ndarray): The Hamiltonian matrix (must be Hermitian).
        time_step (int): The current time step.
        max_steps (int): The maximum number of time steps.
        polar_project (bool): Whether to project the accumulated propagator to the
        unitary space using polar decomposition.
        qr_project (bool): Whether to project the accumulated propagator to the
        unitary space using QR decomposition.

    Returns:
        np.ndarray: The updated accumulated propagator.
    """
    # Ensure the Hamiltonian is Hermitian
    assert np.allclose(
        hamiltonian, hamiltonian.conj().T
    ), "Hamiltonian must be Hermitian."

    # Compute the step propagator
    delta_t = time_step / max_steps
    U_step = expm(-1j * hamiltonian * delta_t)

    # Update the accumulated propagator
    U_accumulated = U_accumulated @ U_step

    # Ensure unitarity by projection if required
    if polar_project and qr_project:
        raise ValueError("Only one of polar_project or qr_project can be True.")
    elif polar_project:
        try:
            U_accumulated, _ = polar(U_accumulated)
        except Exception as e:
            raise RuntimeError(f"Polar decomposition failed: {e}")
    elif qr_project:
        try:
            U_accumulated, _ = qr(U_accumulated)
        except Exception as e:
            raise RuntimeError(f"QR decomposition failed: {e}")

    assert np.allclose(
        U_accumulated @ U_accumulated.conj().T, np.eye(U_accumulated.shape[0])
    ), "The accumulated propagator is not unitary."

    return np.array(U_accumulated, dtype=np.complex128)
