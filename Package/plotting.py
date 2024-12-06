import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
from hyperparameters import config
from matplotlib import cm
from qiskit.visualization import (
    plot_state_city,
    plot_state_qsphere,
    array_to_latex
)
from qutip import Bloch, Qobj
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


def _save_figure(fig, filename):
    """
    Ensures the directory exists and saves the figure to the specified
    filename.
    """
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(filename, dpi=300)
    print(f"\nFigure saved to {filename}.")


def _configure_bloch_sphere_layout(
        bloch,
        sphere_color="#FFDDDD",
        vector_color="g"
        ):
    """
    Configures the layout for a consistent Bloch sphere appearance.
    """
    bloch.vector_color = [vector_color]
    bloch.vector_width = 4
    bloch.sphere_color = sphere_color
    bloch.sphere_alpha = 0.3
    bloch.frame_color = "gray"
    bloch.frame_alpha = 0.2
    bloch.size = [7, 7]
    bloch.view = [-60, 30]
    bloch.xlabel = ["$|+\\rangle$", "$|-\\rangle$"]
    bloch.ylabel = ["$|i\\rangle$", "$|-i\\rangle$"]
    bloch.zlabel = ["$|0\\rangle$", "$|1\\rangle$"]


def initial_state_info(
        initial_state,
        gate,
        fidelity_type,
        basis_type,
        agent_type
        ):
    """
    Displays the initial quantum state information and plots it
    on the Bloch sphere and as a state-city plot.
    """
    dir = config["paths"]["PLOTS"]
    context = f"agent_{agent_type}_gate_{gate}_fidelity_{fidelity_type}_basis_{
        basis_type
        }".replace(
        " ", "_"
    )

    if gate in ["H", "T"]:
        # Plot Bloch sphere
        plot_bloch_sphere_state(
            state=initial_state,
            gate=gate,
            fidelity_type=fidelity_type,
            basis_type=basis_type,
            agent_type=agent_type,
            save=True,
            interactive=False,
        )
    else:
        # Plot state qsphere
        fig = plot_state_qsphere(initial_state)
        filename = dir + f"/q_sphere_state_{context}.png"
        save_figure(fig, filename)
        # plt.close(fig)

    # Plot state city
    fig = plot_state_city(initial_state, figsize=(10, 5), alpha=0.6)
    filename = dir + f"/state_city_{context}.png"
    _save_figure(fig, filename)
    # plt.close(fig)


def plot_results(
    rewards,
    fidelities,
    gate,
    hamiltonian_type,
    agent_type,
    last_update=False,
    save=False,
    interactive=False,
):
    """
    Plots the results of the training process, including rewards,
    fidelities, and control pulse parameters.
    """
    context = f"_Gate_{gate}_Hamiltonian_{hamiltonian_type}_Agent_{
        agent_type}".replace(" ", "_")
    data = [
        (rewards, "Reward per Episode", "Reward"),
        (fidelities, "Fidelity per Episode", "Fidelity"),
        # (omegas, "Rabi Frequency per Episode", "Rabi Frequency", "r"),
        # (deltas, "Detuning per Episode", "Detuning", "g"),
        # (phases, "Phase per Episode", "Phase", "g"),
    ]

    plt.figure(figsize=(15, 20))
    for i, (data_series, label, ylabel, *color) in enumerate(data):
        plt.subplot(len(data), 1, i + 1)
        plt.plot(data_series, label=label, color=color[0] if color else None)
        if not last_update:
            plt.xscale("log")
        plt.xlabel("Episode" if not last_update else "Step")
        plt.ylabel(ylabel)
        plt.legend()
        plt.title(f"{label} ({context})")
        plt.grid()

    plt.tight_layout()
    if save:
        dir = config["paths"]["PLOTS"]
        filename = dir + f"/results_{
            'last_update' if last_update else 'training'
        }{context}.png"
        _save_figure(plt.gcf(), filename)
    if interactive:
        plt.show()
    else:
        plt.close()


# Utility Functions
def _normalize_array(array, label):
    max_val = np.max(np.abs(array))
    if max_val == 0:
        return np.zeros_like(array)
    return array / max_val


def _smooth_array(array, method="none", t_normalized=None, **kwargs):
    if method == "moving_average":
        window_size = kwargs.get("window_size", 5)
        return np.convolve(array, np.ones(
            window_size) / window_size,
            mode="same"
            )
    elif method == "gaussian":
        sigma = kwargs.get("sigma", 2)
        return gaussian_filter1d(array, sigma=sigma)
    elif method == "savitzky_golay":
        window_length = kwargs.get("window_length", 9)
        polyorder = kwargs.get("polyorder", 3)
        return savgol_filter(
            array,
            window_length=window_length,
            polyorder=polyorder
            )
    elif method == "cubic_spline" and t_normalized is not None:
        t_smooth = np.linspace(0, 1, 500)
        spline = CubicSpline(t_normalized, array)
        return spline(t_smooth), t_smooth
    return array, t_normalized


# Single-Qubit Plot
def plot_control_pulse(
    rabi,
    detuning,
    gate,
    hamiltonian_type,
    agent_type,
    smoothing_method="none",
    save=False,
    interactive=False,
    **kwargs,
):
    context = f"Gate_{gate}_Hamiltonian_{hamiltonian_type}_Agent_{
        agent_type}".replace(" ", "_")

    num_steps = len(rabi)
    t_normalized = np.linspace(0, 1, num_steps)

    # Normalize Rabi and Detuning
    rabi_normalized = _normalize_array(rabi, "Rabi")
    detuning_normalized = _normalize_array(detuning, "Detuning")

    # Apply smoothing
    rabi_normalized, t_normalized = _smooth_array(
        rabi_normalized, smoothing_method, t_normalized, **kwargs
    )
    detuning_normalized, t_normalized = _smooth_array(
        detuning_normalized, smoothing_method, t_normalized, **kwargs
    )

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        t_normalized,
        rabi_normalized,
        label=r"$\frac{\mathrm{\Omega}}{\mathrm{\Omega}_{\mathrm{max}}}$",
        color="red",
        linestyle="--",
        drawstyle="steps-post",
    )
    plt.plot(
        t_normalized,
        detuning_normalized,
        label=r"$\frac{\Delta}{\Delta_{\mathrm{max}}}$",
        color="green",
        linestyle="--",
        drawstyle="steps-post",
    )
    plt.xlabel(r"$\frac{t}{T}$", fontsize=14)
    plt.ylabel("Normalized Control Pulses", fontsize=14)
    plt.title("Control Pulses vs Normalized Time", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    # Save or display
    if save:
        dir = config["paths"]["PLOTS"]
        filename = dir + f"/results_last_update_control_pulses_{context}.png"
        _save_figure(plt.gcf(), filename)
    if interactive:
        plt.show()
    else:
        plt.close()


# Two-Qubit Plot
def plot_control_pulses(
    rabi1,
    detuning1,
    rabi2,
    detuning2,
    coupling,
    gate,
    hamiltonian_type,
    agent_type,
    smoothing_method="none",
    save=False,
    interactive=False,
    **kwargs,
):
    context = f"Gate_{gate}_Hamiltonian_{hamiltonian_type}_Agent_{agent_type}".replace(
        " ", "_"
    )
    num_steps = len(rabi1)
    t_normalized = np.linspace(0, 1, num_steps)

    # Normalize all arrays
    rabi1_normalized = _normalize_array(rabi1, "Rabi1")
    detuning1_normalized = _normalize_array(detuning1, "Detuning1")
    rabi2_normalized = _normalize_array(rabi2, "Rabi2")
    detuning2_normalized = _normalize_array(detuning2, "Detuning2")
    coupling_normalized = _normalize_array(coupling, "Coupling")

    # Apply smoothing
    rabi1_normalized, t_normalized = _smooth_array(
        rabi1_normalized, smoothing_method, t_normalized, **kwargs
    )
    detuning1_normalized, t_normalized = _smooth_array(
        detuning1_normalized, smoothing_method, t_normalized, **kwargs
    )
    rabi2_normalized, t_normalized = _smooth_array(
        rabi2_normalized, smoothing_method, t_normalized, **kwargs
    )
    detuning2_normalized, t_normalized = _smooth_array(
        detuning2_normalized, smoothing_method, t_normalized, **kwargs
    )
    coupling_normalized, t_normalized = _smooth_array(
        coupling_normalized, smoothing_method, t_normalized, **kwargs
    )

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        t_normalized,
        rabi1_normalized,
        label=r"$\frac{\mathrm{\Omega_1}}{\mathrm{\Omega}_{\mathrm{1,max}}}$",
        color="red",
        linestyle="--",
        drawstyle="steps-post",
    )
    plt.plot(
        t_normalized,
        detuning1_normalized,
        label=r"$\frac{\Delta_1}{\Delta_{\mathrm{1,max}}}$",
        color="green",
        linestyle="--",
        drawstyle="steps-post",
    )
    plt.plot(
        t_normalized,
        rabi2_normalized,
        label=r"$\frac{\mathrm{\Omega_2}}{\mathrm{\Omega}_{\mathrm{2,max}}}$",
        color="blue",
        linestyle="--",
        drawstyle="steps-post",
    )
    plt.plot(
        t_normalized,
        detuning2_normalized,
        label=r"$\frac{\Delta_2}{\Delta_{\mathrm{2,max}}}$",
        color="orange",
        linestyle="--",
        drawstyle="steps-post",
    )
    plt.plot(
        t_normalized,
        coupling_normalized,
        label=r"$\frac{\mathrm{J}}{\mathrm{J_{max}}}$",
        color="black",
        linestyle="--",
        drawstyle="steps-post",
    )
    plt.xlabel(r"$\frac{t}{T}$", fontsize=14)
    plt.ylabel("Normalized Control Pulses", fontsize=14)
    plt.title("Control Pulses vs Normalized Time", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    # Save or display
    if save:
        dir = config["paths"]["PLOTS"]
        filename = dir + f"/results_last_update_control_pulses_{context}.png"
        _save_figure(plt.gcf(), filename)
    if interactive:
        plt.show()
    else:
        plt.close()


def plot_bloch_sphere_state(
    state,
    gate,
    fidelity_type,
    basis_type,
    agent_type,
    save=False,
    interactive=False,
):
    """
    Plots a single quantum state on the Bloch sphere.
    """
    # Create figure and Bloch sphere and render it
    fig = plt.figure(figsize=(7, 7))
    bloch_sphere = Bloch(fig=fig)
    _configure_bloch_sphere_layout(bloch_sphere)
    bloch_sphere.add_states(Qobj(state))
    bloch_sphere.render()
    if save:
        context = f"agent_{agent_type}_gate_{gate}_fidelity_{
            fidelity_type
            }_basis_{basis_type}".replace(" ", "_")
        dir = config["paths"]["PLOTS"]
        filename = dir + f"/bloch_sphere_state_{context}.png"
        _save_figure(bloch_sphere.fig, filename)
    if interactive:
        bloch_sphere.show()
    else:
        plt.close(fig)


def plot_bloch_sphere_trajectory(
    states,
    gate,
    fidelity_type,
    basis_type,
    agent_type,
    save=False,
    last_update=False,
    interactive=False,
    export_video=False,
):
    """
    Plots the trajectory of quantum states on the Bloch sphere and optionally
    exports a video of the trajectory.
    """
    fig = plt.figure(figsize=(7, 7))
    bloch = Bloch(fig=fig)
    _configure_bloch_sphere_layout(bloch)

    context = f"_Gate_{gate}_Fidelity_{fidelity_type}_Basis_{basis_type}_Hamiltonian_{hamiltonian_type}_Agent_{agent_type}".replace(
        " ", "_"
    )

    # Plot all states
    for state in states:
        bloch.add_states(Qobj(state))
    bloch.render()

    if save:
        dir = config['paths']['PLOTS']
        filename = dir + f"/bloch_sphere_trajectory_{context}_{
            'last_update' if last_update else 'training_full'
            }.png"
        save_figure(bloch.fig, filename)

    # Save video
    if export_video:
        def update(frame):
            bloch.clear()
            bloch.add_states(Qobj(states[frame]))
            bloch.title = f"Frame {frame} ({context})"
            bloch.render()
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(states),
            repeat=False
        )
        writer = animation.FFMpegWriter(fps=5, codec="libx264", bitrate=3000)
        dir = config["paths"]["PLOTS"]
        video_path = dir + f"/bloch_sphere_trajectory_{context}.mp4"
        ani.save(video_path, writer=writer, dpi=300)
        print(f"\nBloch sphere trajectory video saved to {video_path}.")

    if interactive:
        bloch.show()
    else:
        plt.close(fig)


def plot_q_sphere_trajectory(
    states,
    gate,
    fidelity_type,
    basis_type,
    agent_type,
    hamiltonian_type,
    save=False,
    last_update=False,
    interactive=False,
    export_video=False,
):
    """
    Plot the QSphere for a list of quantum states
    and optionally export a video.

    Parameters:
    ----------
    states : list of np.ndarray
        List of quantum states in the form of complex-valued numpy arrays,
        where each state is a 2-element vector [alpha, beta].
    gate : str, optional
        The quantum gate applied (e.g., "H", "T", "CNOT"). Default is None.
    fidelity_type : str, optional
        The type of fidelity calculation used (e.g., "trace", "overlap").
        Default is None.
    basis_type : str, optional
        The type of basis representation (e.g., "computational", "Pauli").
        Default is None.
    agent_type : str, optional
        The type of agent used (e.g., "DQN", "DDPG"). Default is None.
    save : bool, optional
        If True, saves the plot as a PNG file. Default is False.
    last_update : bool, optional
        If True, indicates that the plot is for the last episode of training.
        Default is False.
    interactive : bool, optional
        If True, displays the plot interactively. Default is False.
    export_video : bool, optional
        If True, exports the QSphere trajectory as a video. Default is False.

    Returns:
    ----------
    None: Displays, saves the QSphere plot, or exports a video.
    """
    # Define filenames

    context = f"_Gate_{gate}_Fidelity_{fidelity_type}_Basis_{basis_type}\
    _Hamiltonian_{hamiltonian_type}_Agent_{agent_type}".replace(
        " ", "_"
    )

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw the QSphere (Bloch sphere)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color="r", alpha=0.1, edgecolor="gray")

    # Plot the states
    # Plot the initial state with a distinct color
    initial_state = states[0]
    alpha_init = initial_state[0]
    beta_init = initial_state[1]

    theta_init = 2 * np.arccos(np.abs(alpha_init))  # Polar angle
    phi_init = np.angle(beta_init) - np.angle(alpha_init)  # Azimuthal angle

    x_init = np.sin(theta_init) * np.cos(phi_init)
    y_init = np.sin(theta_init) * np.sin(phi_init)
    z_init = np.cos(theta_init)

    ax.plot(
        [0, x_init],
        [0, y_init],
        [0, z_init],
        color="black",
        linestyle="--"
        )
    ax.scatter(x_init, y_init, z_init, color="black", s=100)

    # Plot the other states
    for idx, state in enumerate(states[1:]):
        alpha = state[0]
        beta = state[1]

        # Calculate spherical coordinates
        theta = 2 * np.arccos(np.abs(alpha))  # Polar angle
        phi = np.angle(beta) - np.angle(alpha)  # Azimuthal angle

        # Cartesian coordinates
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        # Color based on phase
        color = cm.hsv((phi + np.pi) / (2 * np.pi))

        ax.plot([0, x], [0, y], [0, z], color=color)
        ax.scatter(x, y, z, color=color, s=100)

        # Add episode number as a text label
        ax.text(
            x * 1.1,
            y * 1.1,
            z * 1.1,  # Position slightly away from the point
            str(idx),  # Episode or index
            fontsize=8,
            color="black",
        )
    # Add labels and title
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Add labels
    ax.set_xlabel("Re(α)", fontsize=10)
    ax.set_ylabel("Im(α)", fontsize=10)
    ax.set_zlabel("Re(β)", fontsize=10)

    # Construct a descriptive title
    title = "QSphere Representation of Quantum States"
    if gate:
        title += f" | Gate: {gate}"
    if fidelity_type:
        title += f" | Fidelity: {fidelity_type}"
    if basis_type:
        title += f" | Basis: {basis_type}"
    if agent_type:
        title += f" | Agent: {agent_type}"
    if last_update:
        title += " | Last Episode"

    ax.set_title(title, fontsize=12)

    # Save or display the plot
    if save:
        dir = config['paths']['PLOTS']
        filename = dir + f"/q_sphere_trajectory_{context}_{'last_update' if last_update else 'training_full'}.png"
        save_figure(fig, filename)

    if interactive:
        plt.show()
    else:
        plt.close(fig)

    if export_video:
        # Create an animation for the trajectory
        def update(frame):
            ax.clear()
            # Draw the QSphere (Bloch sphere)
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color="r", alpha=0.1, edgecolor="gray")

            for i in range(frame + 1):
                state = states[i]
                alpha = state[0]
                beta = state[1]

                # Calculate spherical coordinates
                theta = 2 * np.arccos(np.abs(alpha))  # Polar angle
                phi = np.angle(beta) - np.angle(alpha)  # Azimuthal angle

                # Cartesian coordinates
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)

                # Color based on phase
                color = cm.hsv((phi + np.pi) / (2 * np.pi))

                # Plot the state
                ax.plot([0, x], [0, y], [0, z], color=color)
                ax.scatter(x, y, z, color=color, s=100)

                # Add index as a label
                ax.text(
                    x * 1.1,
                    y * 1.1,
                    str(i),  # Index or episode number
                    fontsize=8,
                    color="black",
                )

            # Add labels and title
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])

            # Set labels
            ax.set_xlabel("Re(α)")
            ax.set_ylabel("Im(α)")
            ax.set_zlabel("Re(β)")

        # Export the video
        ani = animation.FuncAnimation(
                fig,
                update,
                frames=len(states),
                repeat=False
            )
        writer = animation.FFMpegWriter(fps=5, codec="libx264", bitrate=3000)
        dir = config["paths"]["PLOTS"]
        video_path = dir + f"/q_sphere_trajectory_{context}.mp4"
        ani.save(video_path, writer=writer, dpi=300)
        print(f"\nQSphere trajectory video saved to '{video_path}'.")


def plot_q_sphere(states):
    """
    Plot the QSphere for a list of quantum states.

    Parameters:
    states: list of quantum states in the form of complex-valued numpy arrays
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw the QSphere (Bloch sphere)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color="r", alpha=0.1)

    # Plot the initial state with a distinct color
    initial_state = states[0]
    alpha_init = initial_state[0]
    beta_init = initial_state[1]

    theta_init = 2 * np.arccos(np.abs(alpha_init))  # Polar angle
    phi_init = np.angle(beta_init) - np.angle(alpha_init)  # Azimuthal angle

    x_init = np.sin(theta_init) * np.cos(phi_init)
    y_init = np.sin(theta_init) * np.sin(phi_init)
    z_init = np.cos(theta_init)

    ax.plot([0, x_init], [0, y_init], [0, z_init], color="black",
            linestyle="--")
    ax.scatter(x_init, y_init, z_init, color="black", s=100)

    # Plot the other states
    for idx, state in enumerate(states[1:]):
        alpha = state[0]
        beta = state[1]

        # Calculate spherical coordinates
        theta = 2 * np.arccos(np.abs(alpha))  # Polar angle
        phi = np.angle(beta) - np.angle(alpha)  # Azimuthal angle

        # Cartesian coordinates
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        # Color based on phase
        color = cm.hsv((phi + np.pi) / (2 * np.pi))

        ax.plot([0, x], [0, y], [0, z], color=color)
        ax.scatter(x, y, z, color=color, s=100)

        # Add episode number as a text label
        ax.text(
            x * 1.1,
            y * 1.1,
            str(idx),  # Episode or index
            fontsize=8,
            color="black",
        )

    # Set axis limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Set labels
    ax.set_xlabel("Re(α)")
    ax.set_ylabel("Im(α)")
    ax.set_zlabel("Re(β)")
    plt.show()
