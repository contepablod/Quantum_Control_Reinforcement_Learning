import numpy as np


def gate_fidelity(final_unitary, target_unitary):
    dim = final_unitary.shape[0]  # Dimension of the unitary matrix
    overlap = np.trace(target_unitary.conj().T @ final_unitary)
    fidelity_gate = np.abs(overlap / dim) ** 2
    fidelity_avg = (dim * fidelity_gate + 1) / (dim + 1)
    return fidelity_gate, fidelity_avg


def log_gate_infidelity(final_unitary, target_unitary):
    try:
        fidelity_gate, avg_gate_fidelity = gate_fidelity(
            final_unitary,
            target_unitary
            )
        # Clamp fidelity to avoid log(0) or log(negative)
        # fidelity_gate = max(0, min(fidelity_gate, 1 - 1e-10))
        return (-np.log10(1 - fidelity_gate), fidelity_gate, avg_gate_fidelity)
    except Exception as e:
        print(f"Error calculating infidelity: {e}")
        return None, None, None


def process_average_fidelity(final_unitary, target_unitary):
    # Define the six cardinal Bloch states as density matrices
    bloch_states = [
        np.array([[1, 0], [0, 0]]),  # |0>
        np.array([[0, 0], [0, 1]]),  # |1>
        np.array([[0.5, 0.5], [0.5, 0.5]]),  # |+>
        np.array([[0.5, -0.5], [-0.5, 0.5]]),  # |->
        np.array([[0.5, -0.5j], [0.5j, 0.5]]),  # |i>
        np.array([[0.5, 0.5j], [-0.5j, 0.5]]),  # |-i>
    ]

    fidelity_sum = 0

    # Loop over the six cardinal states
    for state in bloch_states:
        # Apply the target unitary to the initial state
        target_final_state = np.dot(
            target_unitary, np.dot(state, target_unitary.T.conj())
        )
        # Apply the actual unitary to the initial state
        actual_final_state = np.dot(
            final_unitary, np.dot(state, final_unitary.T.conj())
        )
        # Compute fidelity for this state
        fidelity = np.trace(np.dot(
            target_final_state,
            actual_final_state)
        ).real
        fidelity_sum += fidelity

    # Average fidelity over the six Bloch states
    gate_fidelity = fidelity_sum / len(bloch_states)
    return gate_fidelity


def map_average_fidelity(final_map, target_unitary):
    # Define the six cardinal Bloch states as density matrices
    bloch_states = [
        np.array([[1, 0], [0, 0]]),  # |+z>
        np.array([[0, 0], [0, 1]]),  # |-z>
        np.array([[0.5, 0.5], [0.5, 0.5]]),  # |+x>
        np.array([[0.5, -0.5], [-0.5, 0.5]]),  # |-x>
        np.array([[0.5, -0.5j], [0.5j, 0.5]]),  # |+y>
        np.array([[0.5, 0.5j], [-0.5j, 0.5]]),  # |-y>
    ]

    fidelity_sum = 0

    # Loop over the six Bloch states
    for state in bloch_states:
        # Apply the target unitary to the initial state
        target_final_state = np.dot(
            target_unitary, np.dot(state, target_unitary.T.conj())
        )
        # Apply the quantum process (map) to the initial state
        actual_final_state = final_map(state)
        # Compute fidelity for this state
        fidelity = np.trace(np.dot(
            target_final_state,
            actual_final_state)
        ).real
        fidelity_sum += fidelity

    # Average fidelity over the six Bloch states
    map_fidelity = fidelity_sum / len(bloch_states)
    return map_fidelity


def state_fidelity(final_state, target_state):
    # Compute the overlap (inner product) and square the absolute value
    fidelity = np.abs(np.dot(np.conjugate(target_state), final_state)) ** 2
    return fidelity
