from agents import DDDQNAgent
from environment import QuantumGateEnv
from hyperparameters import config
from management_gpu import gpu_management, gpu_info
from plotting import plot_bloch_sphere_state, plot_bloch_sphere_trajectory, plot_results
from qiskit.visualization import plot_state_city, array_to_latex
from training import train_agent
from utils_helpers import compile_model_torch


def main():
    gpu_management()
    gpu_info()

    env = QuantumGateEnv(gate="T")
    agent = DDDQNAgent(env.input_features, env.action_size)
    agent = compile_model_torch(agent)

    print(f"\nThe initial state is: {env.initial_state}\n")
    array_to_latex(env.initial_state, prefix="\\text{Initial State} = ")
    plot_bloch_sphere_state(env.initial_state)
    #fig1.savefig("./Plots/bloch_sphere_state.png")
    plot_state_city(env.initial_state, figsize=(10, 20), alpha=0.6)
    #fig2.savefig("./Plots/state_city.png")

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
            False,
    )

    plot_bloch_sphere_trajectory(state_history)

    plot_results(
        reward_history,
        fidelity_history,
        amplitude_history,
        phase_history,
        duration_history,
        True,
    )

    plot_bloch_sphere_trajectory(env.state_episode)


if __name__ == "__main__":
    main()