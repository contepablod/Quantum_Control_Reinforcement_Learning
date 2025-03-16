import numpy as np
from hyperparameters import config
from numpy.linalg import qr
from scipy.linalg import expm, polar


class Hamiltonian:
    def __init__(self, hamiltonian_type, gate):
        self.hamiltonian_type = hamiltonian_type
        self.gate = gate
        self.X = config["pauli_matrices"]["X_GATE"]
        self.Y = config["pauli_matrices"]["Y_GATE"]
        self.Z = config["pauli_matrices"]["Z_GATE"]
        self.Id = config["pauli_matrices"]["I_GATE"]

    def build_hamiltonian(self, params):
        if self.gate in ["H", "T"]:
            return self.build_one_qubit_hamiltonian(params)
        elif self.gate == "CNOT":
            return self.build_two_qubit_hamiltonian(params)

    def build_one_qubit_hamiltonian(self, params):
        if self.hamiltonian_type == "Field":
            omega = params.get("omega")
            delta = params.get("delta")
            return self._field_hamiltonian(omega, delta)
        elif self.hamiltonian_type == "Geometric":
            omega = params.get("omega")
            delta = params.get("delta")
            phase = params.get("phase")
            return self._geometric_hamiltonian(omega, delta, phase)

    def build_two_qubit_hamiltonian(self, params):
        if self.hamiltonian_type == "Field":
            omega1 = params.get("omega1")
            delta1 = params.get("delta1")
            omega2 = params.get("omega2")
            delta2 = params.get("delta2")
            coupling_strength_zx = params.get("coupling_strength_zx")
            return self._field_hamiltonian_two_qubits(
                omega1, delta1, omega2, delta2, coupling_strength_zx
            )

    # ========================
    # Single-Qubit Hamiltonians
    # ========================

    def _field_hamiltonian(self, omega, delta):
        return omega * self.X + delta * self.Z

    def _geometric_hamiltonian(self, omega, delta, phase):
        return (
            omega * (np.cos(phase) * self.X + np.sin(phase) * self.Y) + delta * self.Z
        )

    # ========================
    # Two-Qubit Hamiltonians
    # ========================

    def _field_hamiltonian_two_qubits(
        self, omega1, delta1, omega2, delta2, coupling_strength_zx
    ):
        # Control qubit Hamiltonian (control field + detuning)
        H_driving_control = omega1 * np.kron(self.X, self.Id)
        # Detuning term for control qubit
        H_detuning_control = delta1 * np.kron(self.Z, self.Id)

        # Target qubit Hamiltonian (control field + detuning)
        H_driving_target = omega2 * np.kron(self.Id, self.X)
        # Detuning term for target qubit
        H_detuning_target = delta2 * np.kron(self.Id, self.Z)

        # Interaction Hamiltonian
        H_interaction_driving = coupling_strength_zx * np.kron(self.Z, self.X)
        # H_interaction_detuning = coupling_strength_zz * np.kron(self.Z, self.Z)

        # Total Hamiltonian
        H_total = (
            H_driving_control
            + H_detuning_control
            + H_driving_target
            + H_detuning_target
            + H_interaction_driving
        )  # + H_interaction_detuning

        return H_total


def update_propagator(
    U_accumulated, hamiltonian, time_delta, polar_project=False, qr_project=False
):

    # Ensure the Hamiltonian is Hermitian
    assert np.allclose(
        hamiltonian, hamiltonian.conj().T
    ), "Hamiltonian must be Hermitian."

    # Compute the step propagator
    U_step = expm(-1j * hamiltonian * time_delta)

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

    # Ensure the accumulated propagator is unitary
    assert np.allclose(
        U_accumulated @ U_accumulated.conj().T,
        np.eye(U_accumulated.shape[0], dtype=np.complex128),
    ), "The accumulated propagator is not unitary."

    return U_accumulated