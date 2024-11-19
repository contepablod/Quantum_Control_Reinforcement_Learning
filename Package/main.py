from agents import DDDQNAgent, DQNAgent
from environment import QuantumGateEnv
from hyperparameters import config
from management_gpu import gpu_management, gpu_info
from plotting import plot_initial_state_info, plot_bloch_sphere_trajectory, plot_results
from training import train_agent
from utils_helpers import compile_model_torch


def main():
    gpu_management()
    gpu_info()

    env = QuantumGateEnv(gate="T")
    agent = DQNAgent(env.input_features, env.action_size)
    agent = compile_model_torch(agent)

    plot_initial_state_info(env.initial_state)

    (
        reward_history,
        fidelity_history,
        state_history,
        amplitude_history,
        phase_history,
        duration_history,
    ) = train_agent(
        agent,
        env,
        config["hyperparameters"]["EPISODES"],
        config["hyperparameters"]["TARGET_UPDATE"],
        config["hyperparameters"]["FIDELITY_THRESHOLD"],
        config["hyperparameters"]["PATIENCE"],
    )

    plot_results(
        reward_history,
        fidelity_history,
        amplitude_history,
        phase_history,
        duration_history,
        last_episode=False,
        save=True,
        interactive=False,
    )

    plot_bloch_sphere_trajectory(state_history, save=True, interactive=False)

    plot_results(
        env.reward_episode,
        env.fidelity_episode,
        env.amplitude_episode,
        env.phase_episode,
        env.duration_episode,
        last_episode=True,
        save=True,
        interactive=False,
    )

    plot_bloch_sphere_trajectory(
        env.state_episode, save=True, interactive=False, export_video=True
    )


if __name__ == "__main__":
    main()
