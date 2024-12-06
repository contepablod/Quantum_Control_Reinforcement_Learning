import os
import shutil
import time
from agents import DDDQNAgent, DQNAgent
from environment import OneQubitEnv, TwoQubitEnv
from hamiltonians import Hamiltonian
from hyperparameters import config
from plotting import (
    # plot_initial_state_info,
    # plot_bloch_sphere_trajectory,
    # plot_control_pulses,
    plot_control_pulse,
    # plot_q_sphere_trajectory,
    plot_results
    )
from pulses import PulseGenerator
from training import Trainer
from utils import (
    gpu_management,
    parse_experiment_arguments,
    compile_model_torch,
    print_hyperparameters
    )


def main():
    # Parse command-line arguments
    args = parse_experiment_arguments()

    # Clean up previous runs
    folder_path = config["paths"]["RUNS"]
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    # GPU Management
    gpu_management()
    device = config["device"]

    # Prepare the environment
    if args:
        gate = args.gate
        hamiltonian_type = args.hamiltonian_type
        control_pulse_type = args.control_pulse_type
        agent_type = args.agent_type
    else:
        gate = 'H'
        hamiltonian_type = 'Field'
        control_pulse_type = "Discrete"
        agent_type = "DDDQN"

    env_classes = {
        "H": OneQubitEnv,
        "T": OneQubitEnv,
        "CNOT": TwoQubitEnv,
    }
    agent_classes = {
        "DQN": DQNAgent,
        "DDDQN": DDDQNAgent
    }

    # Initialize the Hamiltonian
    hamiltonian = Hamiltonian(
        hamiltonian_type=hamiltonian_type,
        gate=gate
    )

    # Initialize the control pulse
    pulse = PulseGenerator(
        control_pulse_type=control_pulse_type,
        gate=gate
    )

    # Initialize the environment
    env = env_classes[gate](
        gate=gate,
        hamiltonian=hamiltonian,
        pulse=pulse,
    )

    # Initialize the agent
    agent = agent_classes[agent_type](
        env=env,
        state_size=env.input_features,
        action_size=env.action_size,
        agent_type=agent_type,
        loss_type=config["hyperparameters"]["LOSS_TYPE"],
        scheduler_type=config["hyperparameters"]["SCHEDULER_TYPE"],
        device=device
    )

    # Display information
    print("\nInformation")
    print(f"\nInitial Propagator: {env.initial_propagator}\n")
    print(f"Gate: {gate}\n")
    print(f"Agent: {agent_type}\n")
    print(f"Hamiltonian: {hamiltonian_type}\n")
    print(f"Control Pulse: {control_pulse_type}\n")
    print("=" * 100)
    print_hyperparameters()

    # Compile the model
    compile_model_torch(agent=agent)

    # Train the agent
    trainer = Trainer(
        agent=agent,
        env=env,
        episodes=config["hyperparameters"]["EPISODES"],
        target_update=config["hyperparameters"]["TARGET_UPDATE"],
        infidelity_threshold=config["hyperparameters"]["INFIDELITY_THRESHOLD"],
        patience=config["hyperparameters"]["PATIENCE"],
    )
    # Start the timer
    start_time = time.time()

    (
        total_reward_history,
        fidelity_history,
    ) = trainer.train()

    # End the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Convert to hours, minutes, and seconds
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Print the training time
    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s.")
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Convert to hours, minutes, and seconds
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Print the training time
    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s.")


    # Plot the total results
    plot_results(
        total_reward_history,
        fidelity_history,
        gate=gate,
        agent_type=agent_type,
        hamiltonian_type=hamiltonian_type,
        last_update=False,
        save=True,
        interactive=False,
    )

    # Plot the results for the last update
    plot_results(
        env.episode_data["discounted_reward"],
        env.episode_data["fidelity"],
        gate=gate,
        agent_type=agent_type,
        hamiltonian_type=hamiltonian_type,
        last_update=True,
        save=True,
        interactive=False,
    )

    plot_control_pulse(
        env.episode_data["control_pulse_params"]["omega"],
        env.episode_data["control_pulse_params"]["delta"],
        gate=gate,
        agent_type=agent_type,
        hamiltonian_type=hamiltonian_type,
        smoothing_method="none",
        save=True,
        interactive=False,
    )

    # plot_control_pulses(
    #     env.omega1_episode,
    #     env.delta1_episode,
    #     env.phase1_episode,
    #     env.omega2_episode,
    #     env.delta2_episode,
    #     env.phase2_episode,
    #     env.coupling_strength_episode,
    #     gate=gate,
    #     agent_type=agent_name,
    #     hamiltonian_type=hamiltonian_type,
    #     save=True,
    #     smoothing_method="none",
    #     interactive=False
    # )

    print('=' * 100, "\n")


# Run the main function
if __name__ == "__main__":
    main()
