import matplotlib.pyplot as plt
import numpy as np
from qutip import Bloch, Qobj

def plot_results(
    rewards, fidelities, amplitudes, phases, durations, last_episode=False
):
    """
    Plots the results of the training process, including rewards,
    fidelities, and control pulse parameters.

    Parameters:
    ----------
    rewards : list
        A list of rewards per episode.
    fidelities : list
        A list of fidelities per episode.
    amplitudes : list
        A list of amplitudes of control pulses per episode.
    phases : list
        A list of phases of control pulses per episode.
    durations : list
        A list of durations of control pulses per episode.
    last_episode : bool, optional
        If True, indicates that the data is from the last episode only.
        In this case, the x-axis will not be logarithmic.
    """
    data = [
        (rewards, "Reward per Episode", "Reward"),
        (fidelities, "Fidelity per Episode", "Fidelity"),
        (amplitudes, "Amplitude per Episode", "Amplitude", "r"),
        (phases, "Phase per Episode", "Phase", "g"),
        (durations, "Duration per Episode", "Duration", "b"),
    ]

    plt.figure(figsize=(15, 20))

    for i, (data_series, label, ylabel, *color) in enumerate(data):
        plt.subplot(5, 1, i + 1)
        plt.plot(data_series, label=label, color=color[0] if color else None)

        # Apply log scale if not last episode and data is long enough
        if not last_episode:
            plt.xscale("log")

        plt.xlabel("Episode" if not last_episode else "Step")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.show()


def plot_bloch_sphere_state(state):
    """
    Plots the quantum state on the Bloch sphere.

    Parameters:
    ----------
    state (np.ndarray): The quantum state vector [alpha, beta] where
                        alpha is the coefficient of |0> and beta is the coefficient of |1>.
    """
    # Check if the state is normalized
    norm = np.linalg.norm(state)
    if not np.isclose(norm, 1.0):
        print(
            f"Warning: The state is not normalized (norm = {norm:.4f}). Normalizing the state."
        )
        # Normalize the state
        state = state / norm

    # Extract the coefficients
    alpha = state[0]
    beta = state[1]

    # Calculate the Bloch sphere coordinates
    x = 2 * np.real(np.conj(alpha) * beta)
    y = 2 * np.imag(np.conj(alpha) * beta)
    z = np.abs(alpha) ** 2 - np.abs(beta) ** 2
    bloch_vector = [x, y, z]

    # Create Bloch sphere instance with customization
    bloch_sphere = Bloch()
    bloch_sphere.add_vectors(bloch_vector)
    bloch_sphere.vector_color = ["g"]  # Set vector color
    bloch_sphere.vector_width = 4  # Set vector width
    bloch_sphere.sphere_color = "#FFDDDD"  # Set Bloch sphere color
    bloch_sphere.sphere_alpha = 0.3  # Set sphere transparency
    bloch_sphere.frame_color = "gray"  # Set wireframe color
    bloch_sphere.frame_alpha = 0.2  # Set wireframe transparency
    bloch_sphere.size = [7, 7]  # Set figure size (700x700 pixels)
    bloch_sphere.view = [-60, 30]  # Set viewing angles

    # Set axis labels
    bloch_sphere.xlabel = ["$|+\\rangle$", "$|-\\rangle$"]
    bloch_sphere.ylabel = ["$|i\\rangle$", "$|-i\\rangle$"]
    bloch_sphere.zlabel = ["$|0\\rangle$", "$|1\\rangle$"]

    # Display the Bloch sphere
    bloch_sphere.show()

    # Return the figure object for saving
    return bloch_sphere.fig


def plot_bloch_sphere_trajectory(states):
    """
    Plot a trajectory of quantum states on the Bloch sphere.

    Parameters:
    - states (list of np.ndarray): A list of quantum state vectors.
      Each state should be a 2-element complex numpy array representing a qubit state.

    Returns:
    - None: Displays the Bloch sphere plot with the trajectory of the given states.
    """
    # Initialize Bloch sphere
    bloch = Bloch()

    # Add each state to the Bloch sphere
    for state in states:
        # Convert the state to a Qobj (quantum object) for compatibility with QuTiP
        bloch.add_states(Qobj(state))

    # Display the Bloch sphere
    bloch.show()
    return bloch.fig
