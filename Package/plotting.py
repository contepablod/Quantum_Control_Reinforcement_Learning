import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from hyperparameters import config
from qiskit.visualization import plot_state_city, array_to_latex
from qutip import Bloch, Qobj


def plot_results(
    rewards,
    fidelities,
    amplitudes,
    phases,
    durations,
    last_episode=False,
    save=False,
    interactive=False,
):
    """
    Plots the results of the training process, including rewards,
    fidelities, and control pulse parameters.

    Parameters:
    - rewards (list): List of rewards per episode.
    - fidelities (list): List of fidelities per episode.
    - amplitudes (list): List of amplitudes per episode.
    - phases (list): List of phases per episode.
    - durations (list): List of durations per episode.
    - last_episode (bool, optional): If True, plots data for the last episode.
    Default is False.
    - save (bool, optional): If True, saves the plots as PNG files.
    Default is False.
    - interactive (bool, optional): If True, displays the plots interactively.
    Default is False.

    Returns:
    - None: Displays or saves the plots.
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

    # Save the figure
    if save:
        if last_episode:
            plt.savefig("../Plots/last_episode_results.png")
            print("\nPlot saved to '../Plots/last_episode_results.png'.")
        else:
            plt.savefig("../Plots/results.png")
            print("\nPlot saved to '../Plots/results.png'.")

    # Display or close the figure
    if interactive:
        plt.show()
    else:
        plt.close()  # Close the current figure to free memory


def plot_initial_state_info(initial_state):
    """
    Displays the initial quantum state information and plots it
    on the Bloch sphere.

    Parameters:
    - initial_state (np.ndarray): A 2-element complex numpy array representing
      the initial quantum state vector [alpha, beta], where alpha is the
      coefficient of |0⟩ and beta is the coefficient of |1⟩.

    Returns:
    - None: Prints the initial state, displays it in LaTeX format,
      and saves the Bloch sphere plot and state city plot.
    """
    print(f"\nThe initial state is: {initial_state}")
    array_to_latex(initial_state, prefix="\\text{Initial State} = ")

    # Plot and save the Bloch sphere representation
    plot_bloch_sphere_state(initial_state, save=True)

    # Create the state city plot
    fig = plot_state_city(initial_state, figsize=(10, 20), alpha=0.6)

    # Save the state city plot
    fig.savefig("../Plots/state_city.png")
    print("\nState city plot saved to '../Plots/state_city.png'.\n")

    # Close the figure to free up memory
    plt.close(fig)


def plot_bloch_sphere_state(state, save=False, interactive=False):
    """
    Plots a single quantum state on the Bloch sphere.

    Parameters:
    - state (np.ndarray): A 2-element complex numpy array representing
    the quantum state vector.
    - save (bool, optional): If True, saves the plot as a PNG file.
    Default is False.
    - interactive (bool, optional): If True, displays the plot interactively.
    Default is False.

    Returns:
    - None: Displays the Bloch sphere plot of the given state.
    """
    # Check if the state is normalized
    norm = np.linalg.norm(state)
    if not np.isclose(norm, 1.0):
        print(
            f"Warning: The state is not normalized (norm = {norm:.4f}). \
            Normalizing the state."
        )
        state = state / norm

    # Calculate the Bloch sphere coordinates
    alpha = state[0]
    beta = state[1]
    x = 2 * np.real(np.conj(alpha) * beta)
    y = 2 * np.imag(np.conj(alpha) * beta)
    z = np.abs(alpha) ** 2 - np.abs(beta) ** 2
    bloch_vector = [x, y, z]

    # Create figure and Bloch sphere instance
    fig = plt.figure(figsize=(7, 7))
    bloch_sphere = Bloch(fig=fig)
    _configure_bloch_sphere_layout(bloch_sphere)
    bloch_sphere.add_vectors(bloch_vector)

    # Render and save/display the Bloch sphere
    bloch_sphere.render()
    if save:
        bloch_sphere.fig.savefig("../Plots/bloch_sphere.png")
        print("\nBloch sphere plot saved to '../Plots/bloch_sphere.png'.")
    if interactive:
        bloch_sphere.show()
    else:
        plt.close(fig)


def plot_bloch_sphere_trajectory(
    states, save=False, interactive=False, export_video=False
):
    """
    Plots the trajectory of quantum states on the Bloch sphere and optionally
    exports a video of the trajectory.

    Parameters:
    - states (list of np.ndarray): A list of 2-element complex numpy
      arrays representing the quantum state vectors at different time steps.
    - save (bool, optional): If True, saves the plot as a PNG file.
      Default is False.
    - interactive (bool, optional): If True, displays the plot interactively.
      Default is False.
    - export_video (bool, optional): If True, exports the trajectory
    as a video.
      Default is False.

    Returns:
    - None: Displays the Bloch sphere plot with the trajectory
      of the given states or exports a video.
    """
    # Initialize figure and Bloch sphere
    fig = plt.figure(figsize=(7, 7))
    bloch = Bloch(fig=fig)
    _configure_bloch_sphere_layout(bloch)

    # Add each state to the Bloch sphere
    def update(frame):
        bloch.clear()
        bloch.add_states(Qobj(states[frame]))
        bloch.render()

    if export_video:
        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=len(states), repeat=False)
        # Export video using ffmpeg
        video_path = "../Plots/bloch_sphere_trajectory.mp4"
        ani.save(video_path, writer="ffmpeg", fps=10)
        print(f"\nBloch sphere trajectory video saved to {video_path}.")

    else:
        # Render the Bloch sphere and save/display
        for state in states:
            bloch.add_states(Qobj(state))
        bloch.render()
        if save:
            bloch.fig.savefig("../Plots/bloch_sphere_trajectory.png")
            print(
                "\nBloch sphere trajectory saved to \
                '../Plots/bloch_sphere_trajectory.png'."
            )
        if interactive:
            bloch.show()
        else:
            plt.close(fig)


def _configure_bloch_sphere_layout(bloch):
    """
    Configures the layout for a consistent Bloch sphere appearance.

    Parameters:
    - bloch (qutip.Bloch): The Bloch sphere instance to configure.

    Returns:
    - None: Applies styling to the given Bloch sphere.
    """
    bloch.vector_color = ["g"]
    bloch.vector_width = 4
    bloch.sphere_color = "#FFDDDD"
    bloch.sphere_alpha = 0.3
    bloch.frame_color = "gray"
    bloch.frame_alpha = 0.2
    bloch.size = [7, 7]
    bloch.view = [-60, 30]
    bloch.xlabel = ["$|+\\rangle$", "$|-\\rangle$"]
    bloch.ylabel = ["$|i\\rangle$", "$|-i\\rangle$"]
    bloch.zlabel = ["$|0\\rangle$", "$|1\\rangle$"]
