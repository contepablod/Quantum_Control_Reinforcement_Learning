import numpy as np


def state_fidelity(final_state, target_state):
    """
    Calculates the state fidelity between the final state and the target state.

    This function measures the overlap between two pure quantum states.
    It returns a value between 0 and 1, where 1 indicates that
    the states are identical.

    Parameters:
    ----------
    final_state (np.ndarray): The final quantum state as
    a complex vector (1D array).
    target_state (np.ndarray): The target quantum state as
    a complex vector (1D array).

    Returns:
    -------
    float
        The fidelity between the final state and the target state.
    """
    # Compute the overlap (inner product) and square the absolute value
    fidelity = np.abs(np.dot(np.conjugate(target_state), final_state)) ** 2
    return fidelity


def gate_fidelity(final_unitary, target_unitary):
    """
    Calculate the fidelity between the accumulated propagator and the target unitary.

    Parameters:
        U_accumulated (np.ndarray): The accumulated propagator matrix (unitary).
        U_target (np.ndarray): The target unitary matrix.

    Returns:
        float: The fidelity value.
    """

    dim = final_unitary.shape[0]  # Dimension of the unitary matrix

    # Fidelity calculation
    overlap = np.trace(np.dot(target_unitary.conj().T, final_unitary))
    fidelity = np.abs(overlap/dim) ** 2
    #return np.clip(fidelity, 0, 1)
    return fidelity


def gate_average_fidelity(final_unitary, target_unitary):
    """
    Calculates the gate fidelity between the final unitary operation and
    the target unitary.

    Gate fidelity is computed as an average over the six cardinal Bloch states:
    |+>, |i>, |i>, |->, |0>, and |1>. It measures how well
    the implemented gate approximates the target unitary operation.

    Parameters:
    ----------
    final_unitary (np.ndarray): The unitary matrix representing
    the operation performed by the control process.
    target_unitary (np.ndarray): The ideal unitary matrix representing
    the target gate (e.g., Hadamard gate).

    Returns:
    -------
    float
        The gate fidelity, a value between 0 and 1,
        where 1 indicates perfect fidelity.
    """
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


def map_fidelity(final_map, target_unitary):
    """
    Calculates the map (or process) fidelity between a general quantum process
    (map) and an ideal unitary operation.

    This function compares a general quantum process, which may be noisy or
    non-unitary, against the ideal target unitary by averaging the fidelity
    over a set of representative input states (the six cardinal Bloch states).
    It measures how closely the process approximates the target operation.

    Parameters:
    ----------
    final_map (function): A function representing the quantum process
    (superoperator) to be evaluated. It should accept a density matrix
    as input and return a transformed density matrix.
    target_unitary (np.ndarray): The ideal unitary matrix representing
    the target operation.

    Returns:
    -------
    float
        The map fidelity, a value between 0 and 1, where 1 indicates
        perfect fidelity.
    """
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
