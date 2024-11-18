import numpy as np


def generate_state(num_qubits, basis="random"):
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


# Example usage:
num_qubits = 2
state_random = generate_state(num_qubits, basis="random")
state_z_basis = generate_state(num_qubits, basis="Z")
state_x_basis = generate_state(num_qubits, basis="X")
state_y_basis = generate_state(num_qubits, basis="Y")

print(f"Random state for {num_qubits} qubits:\n", state_random)
print("Norm of the random state:", np.linalg.norm(state_random))

print(f"Z-basis state for {num_qubits} qubits:\n", state_z_basis)
print("Norm of the Z-basis state:", np.linalg.norm(state_z_basis))

print(f"X-basis (Hadamard) state for {num_qubits} qubits:\n", state_x_basis)
print("Norm of the X-basis state:", np.linalg.norm(state_x_basis))

print(f"Y-basis state for {num_qubits} qubits:\n", state_y_basis)
print("Norm of the Y-basis state:", np.linalg.norm(state_y_basis))
