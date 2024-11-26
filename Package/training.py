from torch.utils.tensorboard.writer import SummaryWriter
from torch.cuda import empty_cache
from tqdm import trange
import numpy as np


class Trainer:
    """
    Trainer class for reinforcement learning agents
    with TensorBoard integration.
    """

    def __init__(
            self,
            agent,
            env,
            episodes,
            target_update,
            infidelity_threshold,
            patience,
            log_dir="runs"
            ):
        self.agent = agent
        self.env = env
        self.episodes = episodes
        self.target_update = target_update
        self.infidelity_threshold = infidelity_threshold
        self.patience = patience

        # Tracking variables
        self.reward_history = []
        self.fidelity_history = []
        self.state_history = []
        self.amplitude_history = []
        self.phase_history = []
        self.duration_history = []

        # Early stopping variables
        self.best_log_infidelity = 0
        self.best_fidelity = 0
        self.patience_counter = 0

        # TensorBoard Writer
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self):
        """
        Trains the agent over the specified number of episodes
        with TensorBoard logging.
        """
        for e in trange(self.episodes, desc="Training Episodes"):
            # Run episode
            total_reward = self.run_episode(e)

            # Train the agent
            loss = self.agent.replay()

            # Log loss if available
            if loss is not None:
                self.writer.add_scalar("Loss", loss, e)

            # Log learning rate
            current_lr = self.agent.lr_scheduler.get_last_lr()[0]
            self.writer.add_scalar("Learning Rate", current_lr, e)

            # # Log histograms for model weights and biases
            # self._log_model_histograms(e)

            # Update target model periodically
            if e > 0 and e % self.target_update == 0:
                self.agent.update_target_model()

            # Log metrics
            self.writer.add_scalar("Reward", total_reward, e)
            self.writer.add_scalar("Fidelity", self.fidelity_history[-1], e)
            self.writer.add_scalar("Epsilon", self.agent.epsilon, e)

            # Check for early stopping
            if self.check_early_stopping(total_reward, e):
                break

            # Close the TensorBoard writer
            self.writer.close()

            # Free GPU memory
            empty_cache()

        # Close the TensorBoard writer
        self.writer.close()

        print("Training finished.")

        return (
            self.reward_history,
            self.fidelity_history,
            self.state_history,
            self.amplitude_history,
            self.phase_history,
            self.duration_history,
        )

    def run_episode(self, episode):
        """
        Runs a single episode of training.

        Parameters:
        ----------
        episode : int
            The current episode number.

        Returns:
        -------
        float
            The total reward for the episode.
        """
        state = self.env.reset()
        total_reward = 0

        for _ in range(self.env.max_steps):
            action = self.agent.act(state)
            (
                done,
                next_state,
                amplitude, phase,
                duration,
                reward,
                fidelity
            ) = self.env.step(action)

            # Remember experience
            self.agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        # Record metrics
        self.state_history.append(state)
        self.amplitude_history.append(amplitude)
        self.phase_history.append(phase)
        self.duration_history.append(duration)
        self.reward_history.append(total_reward)
        self.fidelity_history.append(fidelity)

        # Log episode-specific metrics
        self.writer.add_scalar("Amplitude", np.mean(amplitude), episode)
        self.writer.add_scalar("Phase", np.mean(phase), episode)
        self.writer.add_scalar("Duration", np.mean(duration), episode)

        # Print progress
        if episode > 0 and episode % (self.episodes // 100) == 0:
            print(
                f"\tEpisode: {episode}/{self.episodes}",
                f"Reward: {total_reward:.5f}",
                f"Fidelity: {fidelity:.5f}",
                f"Current Epsilon: {self.agent.epsilon:.5f}",
                f"Current LR: {self.agent.lr_scheduler.get_last_lr()[0]:.5f}",
                f"Patience: {self.patience_counter}\n"
            )

        return total_reward

    def check_early_stopping(self, total_reward, episode):
        """
        Checks if early stopping conditions are met.

        Parameters:
        ----------
        total_reward : float
            The total reward achieved in the current episode.
        episode : int
            The current episode number.

        Returns:
        -------
        bool
            True if early stopping conditions are met, False otherwise.
        """
        current_log_infidelity, current_fidelity = self.env.log_infidelity(
            self.state_history[-1]
        )

        if current_log_infidelity >= self.best_log_infidelity:
            self.best_log_infidelity = current_log_infidelity
            self.best_fidelity = current_fidelity
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        self.writer.add_scalar(
            "Patience Counter",
            self.patience_counter,
            episode
            )

        if -self.best_log_infidelity >= self.infidelity_threshold \
                or self.patience_counter >= self.patience:
            print(
                "Early stopping triggered.",
                f"Achieved fidelity: {self.best_fidelity:.5f}",
                f"Episode: {episode}, Patience: {self.patience_counter}",
                f"Achieved Reward: {total_reward:.5f}",
                f"Final Epsilon: {self.agent.epsilon:.5f}",
                f"Final LR: {self.agent.lr_scheduler.get_last_lr()[0]:.5f}"
            )
            return True

        return False

    # def _log_model_histograms(self, episode):
    #     """
    #     Logs histograms of model weights and gradients to TensorBoard.

    #     Parameters:
    #     ----------
    #     episode : int
    #         The current episode number.
    #     """
    #     for name, param in self.agent.model.named_parameters():
    #         if param.requires_grad:
    #             # Log weights
    #             self.writer.add_histogram(
    #                 f"Weights/{name}", param.data.cpu().numpy(), episode
    #             )
    #             # Log gradients only if they exist
    #             if param.grad is not None:
    #                 self.writer.add_histogram(
    #                     f"Gradients/{name}", param.grad.cpu().numpy(), episode
    #                 )
