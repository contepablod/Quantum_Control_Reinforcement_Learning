import os
import shutil
import time
from q_agents import DDDQNAgent
from p_agents import PPOAgent
from t_agents import TD3Agent
from gp_agents import GPAgent
from environment import GateEnv
from hamiltonians import Hamiltonian
from hyperparameters import config
from pulses import PulseGenerator
from q_training import QTrainer
from p_training import PTrainer
from t_training import TTrainer
from gp_training import GPTrainer
from torch import compile, __version__ as torch_version, _dynamo as torch_dynamo
from utils import (
    get_execution_time,
    gpu_management,
    parse_experiment_arguments,
    print_hyperparameters,
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
        gate = "H"
        hamiltonian_type = "Field"
        control_pulse_type = "Discrete"
        agent_type = "DDDQN"

    agent_classes = {
        "DDDQN": DDDQNAgent,
        "PPO": PPOAgent,
        "GP": GPAgent,
        "TD3": TD3Agent,
    }

    # Initialize the Hamiltonian
    hamiltonian = Hamiltonian(hamiltonian_type=hamiltonian_type, gate=gate)

    # Initialize the control pulse
    pulse = PulseGenerator(
        control_pulse_type=control_pulse_type,
        gate=gate,
        hamiltonian_type=hamiltonian_type,
        agent_type=agent_type,
    )

    # Initialize the environment
    env = GateEnv(
        gate=gate,
        hamiltonian=hamiltonian,
        pulse=pulse,
        device=device,
    )

    # Initialize the agent
    agent = agent_classes[agent_type](
        env=env,
        agent_type=agent_type,
        device=device,
        loss_type=config["hyperparameters"]["loss"]["LOSS_TYPE"],
    )

    # Display information
    print("ℹ️  Information")
    print(f"\nInitial Propagator: {env.initial_propagator}\n")
    print(f"Gate: {gate}\n")
    print(f"Agent: {agent_type}\n")
    print(f"Hamiltonian: {hamiltonian_type}\n")
    print(f"Control Pulse: {control_pulse_type}")
    print("=" * 100)
    print_hyperparameters()

    # Compile the model
    if torch_version >= "2.0.0":
        torch_dynamo.reset()
        print("🛠️  Compiling model...")
        print("=" * 100)
        if agent_type == "PPO":
            # agent.actor = compile(
            #     model=agent.actor,
            #     fullgraph=True,
            #     dynamic=True,
            #     mode="max-autotune",
            # )
            # agent.critic = compile(
            #         model=agent.critic,
            #         fullgraph=True,
            #         dynamic=True,
            #         mode="max-autotune",
            # )
            # agent.old_actor = compile(
            #         model=agent.old_actor,
            #         fullgraph=True,
            #         dynamic=True,
            #         mode="max-autotune",
            # )
            agent.ppo = compile(
                model=agent.ppo,
                fullgraph=True,
                dynamic=True,
                mode="reduce-overhead",
            )
        elif agent_type == "DDDQN":
            agent.model = compile(
                model=agent.model,
                fullgraph=True,
                dynamic=True,
                mode="reduce-overhead",
            )
            agent.target_model = compile(
                model=agent.target_model,
                fullgraph=True,
                dynamic=True,
                mode="reduce-overhead",
            )
        elif agent_type == "TD3":
            agent.actor = compile(
                model=agent.actor,
                fullgraph=True,
                dynamic=True,
                mode="reduce-overhead",
            )
            agent.critic_1 = compile(
                model=agent.critic_1,
                fullgraph=True,
                dynamic=True,
                mode="reduce-overhead",
            )
            agent.critic_2 = compile(
                model=agent.critic_2,
                fullgraph=True,
                dynamic=True,
                mode="reduce-overhead",
            )
            # agent.critic_1_target = compile(
            #     model=agent.critic_1_target,
            #     fullgraph=True,
            #     dynamic=True,
            #     mode="reduce-overhead",
            # )
            # agent.critic_2_target = compile(
            #     model=agent.critic_2_target,
            #     fullgraph=True,
            #     dynamic=True,
            #     mode="reduce-overhead",
            # )
        elif agent_type == "GP":
            agent.model = compile(
                model=agent.grpo,
                fullgraph=True,
                dynamic=True,
                mode="reduce-overhead",
            )

    # Map trainer class
    trainer_classes = {
        "DDDQN": QTrainer,
        "PPO": PTrainer,
        "TD3": TTrainer,
        "GP": GPTrainer,
    }

    # Train the agent
    trainer = trainer_classes[agent_type](
        agent=agent,
        env=env,
        episodes=config["hyperparameters"]["train"]["EPISODES"],
        patience=config["hyperparameters"]["train"]["PATIENCE"],
        device=device,
        save_final_model=True,
        model_dir=config["paths"]["MODELS"],
        save_metrics=True,
        metrics_dir=config["paths"]["DATA"],
        metrics_format=config["paths"]["METRICS_FORMAT"],
    )

    # Start the timer
    start_time = time.time()

    # Train the agent
    trainer.train()

    # End the timer
    end_time = time.time()

    # Calculate the execution time
    get_execution_time(start_time, end_time)

    print("=" * 100, "\n")


if __name__ == "__main__":
    main()
