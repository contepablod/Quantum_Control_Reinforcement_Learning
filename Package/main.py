import argparse
import os
import shutil
from agents import DDDQNAgent
from environment import QuantumEnv
from hyperparameters import config
from plotting import (
    plot_initial_state_info,
    plot_bloch_sphere_trajectory,
    plot_q_sphere_trajectory,
    plot_results
    )
from training import Trainer
from utils import gpu_management, compile_model_torch


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Quantum Control Experiment Parameters"
    )

    # Add optional arguments with default values
    parser.add_argument(
        "--gate",
        type=str,
        default="H",
        help="Gate type (e.g., H, T or CNOT) [default: H]",
    )
    parser.add_argument(
        "--fidelity_type",
        type=str,
        default="state",
        help="Fidelity type (e.g., state, gate) [default: state]",
    )
    parser.add_argument(
        "--basis_type",
        type=str,
        default="Z",
        help="Basis type (e.g., Z, X, etc.) [default: Z]",
    )
    parser.add_argument(
        "--hamiltonian_type",
        type=str,
        default="field_driven",
        help="Hamiltonian type (e.g., field_driven, rotational, etc.) \
        [default: field_driven]",
    )

    args = parser.parse_args()

    # Deletes folder and all its contents
    folder_path = "/home/pdconte/Desktop/DUTh_Thesis/Package/runs"
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    # GPU Management
    gpu_management()
    device = config["device"]

    # Prepare the environment
    if args:
        gate = args.gate
        fidelity_type = args.fidelity_type
        basis_type = args.basis_type
        hamiltonian_type = args.hamiltonian_type
    else:
        gate = 'H'
        fidelity_type = 'state'
        basis_type = 'Z'
        hamiltonian_type = 'field_driven'

    # Initialize the environment
    env = QuantumEnv(
        gate=gate,
        fidelity_type=fidelity_type,
        basis_type=basis_type,
        hamiltonian_type=hamiltonian_type
    )

    # Initialize the agent
    agent = DDDQNAgent(
        state_size=env.input_features,
        action_size=env.action_size,
        loss_type=config["hyperparameters"]["LOSS_TYPE"],
        scheduler_type=config["hyperparameters"]["SCHEDULER_TYPE"],
        device=device


    )

    # Compile the model
    agent_name = agent.__class__.__name__
    agent = compile_model_torch(agent=agent)

    # Plot the initial state
    plot_initial_state_info(
        env.initial_state,
        gate,
        fidelity_type,
        basis_type,
        agent_name
    )

    # Train the agent
    trainer = Trainer(
        agent=agent,
        env=env,
        episodes=config["hyperparameters"]["EPISODES"],
        target_update=config["hyperparameters"]["TARGET_UPDATE"],
        infidelity_threshold=config["hyperparameters"]["INFIDELITY_THRESHOLD"],
        patience=config["hyperparameters"]["PATIENCE"],
    )

    (
        reward_history,
        fidelity_history,
        state_history,
        amplitude_history,
        phase_history,
        duration_history
    ) = trainer.train()

    # Plot the results for all episodes
    plot_results(
        reward_history,
        fidelity_history,
        amplitude_history,
        phase_history,
        duration_history,
        gate=gate,
        fidelity_type=fidelity_type,
        basis_type=basis_type,
        agent_type=agent_name,
        last_update=False,
        save=True,
        interactive=False,
    )

    if gate in ['H', 'T']:
        plot_bloch_sphere_trajectory(
            states=state_history,
            gate=gate,
            fidelity_type=fidelity_type,
            basis_type=basis_type,
            agent_type=agent_name,
            save=True,
            last_update=False,
            interactive=False,
            export_video=False,
        )
    else:
        plot_q_sphere_trajectory(
            states=state_history,
            gate=gate,
            fidelity_type=fidelity_type,
            basis_type=basis_type,
            agent_type=agent_name,
            save=True,
            last_update=False,
            interactive=False,
            export_video=False,
        )

    # Plot the results for the last update
    plot_results(
        env.reward_episode,
        env.fidelity_episode,
        env.amplitude_episode,
        env.phase_episode,
        env.duration_episode,
        gate=gate,
        fidelity_type=fidelity_type,
        basis_type=basis_type,
        agent_type=agent_name,
        last_update=True,
        save=True,
        interactive=False,
    )

    if gate in ['H', 'T']:
        plot_bloch_sphere_trajectory(
            states=env.state_episode,
            gate=gate,
            fidelity_type=fidelity_type,
            basis_type=basis_type,
            agent_type=agent_name,
            save=True,
            last_update=True,
            interactive=False,
            export_video=True,
        )
    else:
        plot_q_sphere_trajectory(
            states=env.state_episode,
            gate=gate,
            fidelity_type=fidelity_type,
            basis_type=basis_type,
            agent_type=agent_name,
            save=True,
            last_update=True,
            interactive=False,
            export_video=True,
        )


# Run the main function
if __name__ == "__main__":
    main()
