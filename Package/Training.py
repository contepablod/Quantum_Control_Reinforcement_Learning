# Training the agent
def train_agent(agent, env, episodes, target_update, fidelity_threshold, patience):
    """
    Trains a reinforcement learning agent in a specified environment over a series of episodes.

    Parameters:
    -----------
    agent : object
        The reinforcement learning agent to be trained. The agent must have the methods `act(state)`, `remember(state, action, reward, next_state, done)`, `replay()`, and `update_target_model()` implemented.
    env : object
        The environment in which the agent is trained. The environment must have the methods `reset()`, `step(action)`, and `infidelity(state)` implemented, as well as the attribute `max_steps`.
    episodes : int
        The number of training episodes.
    target_update : int
        The frequency (in episodes) at which the agent's target model is updated.
    fidelity_threshold : float
        The fidelity threshold for early stopping. Training will stop early if the best fidelity achieved is greater than or equal to `1 - fidelity_threshold`.
    patience : int
        The patience parameter for early stopping. If fidelity does not improve for this many consecutive episodes, training will stop early.

    Returns:
    --------
    total_rewards : list of float
        A list containing the total rewards obtained in each episode.
    fidelities : list of float
        A list containing the fidelities achieved in each episode, where fidelity is defined as `1 - infidelity`.
    state_history : list
        A list of final states from each episode.
    amplitudes_history : list
        A list containing the amplitudes recorded at each episode step.
    phases_history : list
        A list containing the phases recorded at each episode step.
    durations_history : list
        A list containing the durations recorded at each episode step.

    Notes:
    ------
    - Early stopping is triggered if the best fidelity exceeds `1 - fidelity_threshold` or if the fidelity does not improve for `patience` consecutive episodes.
    - The method assumes that the environment's `step` method returns `next_state, reward, done, amplitudes, phases, durations`.
    """

    reward_history = []
    fidelity_history = []
    state_history = []
    amplitude_history = []
    phase_history = []
    duration_history = []
    best_fidelity = 0
    patience_counter = 0

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(env.max_steps):
            action = agent.act(state)
            (done, next_state, amplitude, phase, duration, reward, fidelity) = env.step(
                action
            )
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        agent.replay()

        if e % target_update == 0:
            agent.update_target_model()

        state_history.append(state)
        amplitude_history.append(amplitude)
        phase_history.append(phase)
        duration_history.append(duration)
        reward_history.append(total_reward)
        fidelity_history.append(fidelity)

        if e % (EPISODES / 100) == 0:
            print(
                f"Episode: {e}/{episodes},  Reward: {total_reward:.5f}, "
                f"Fidelity: {fidelity_history[-1]:.5f},  Epsilon: {agent.epsilon:.5f}"
            )
        # Early stopping check
        current_fidelity = 1 - env.infidelity(state)
        if current_fidelity >= best_fidelity:
            best_fidelity = current_fidelity
            patience_counter = 0  # Reset patience counter if fidelity improves
        else:
            patience_counter += 1

        if best_fidelity >= (1 - fidelity_threshold) or patience_counter >= patience:
            print(
                f"Early stopping triggered. Achieved fidelity: {best_fidelity:.5f}, "
                f"Episode: {e}, Patience: {patience_counter}, Reward: {total_reward:.5f}, "
                f"Epsilon: {agent.epsilon:.5f}"
            )
            break

    print("Training finished.")
    return (
        reward_history,
        fidelity_history,
        state_history,
        amplitude_history,
        phase_history,
        duration_history,
    )
