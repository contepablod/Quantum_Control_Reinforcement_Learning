import numpy as np
import random
from hyperparameters import config
from torch.cuda import empty_cache
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import trange


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
            log_dir="runs",
            initial_replay_buffer=False
            ):
        """
        Initializes a Trainer object to train the specified agent in the
        given environment.

        Parameters
        ----------
        agent : agents.BaseQAgent
            The agent to be trained. Must be an instance of a subclass of
            BaseQAgent.

        env : environments.QuantumEnv
            The environment in which the agent is to be trained. Must be an
            instance of QuantumEnv.

        episodes : int
            The number of episodes to train the agent for.

        target_update : int
            The number of episodes between each target model update.

        infidelity_threshold : float
            The minimum acceptable infidelity value for the agent to be
            considered as having converged.

        patience : int
            The number of episodes to wait before stopping training if the
            agent has not converged.

        log_dir : str, optional
            The directory to log the training to using TensorBoard. Defaults
            to "runs".
        """
        self.agent = agent
        self.env = env
        self.episodes = episodes
        self.target_update = target_update
        self.infidelity_threshold = infidelity_threshold
        self.patience = patience
        self.initial_replay_buffer = initial_replay_buffer

        # Initialize a metrics dictionary
        self.metrics = {
            "episode": [],
            "total_reward": [],
            "discounted_reward": [],
            "fidelity": [],
            "log_infidelity": [],
            "state": [],
        }

        # Early stopping variables
        self.best_log_infidelity = 0
        self.best_fidelity = 0
        self.patience_counter = 0

        # TensorBoard Writer
        self.writer = SummaryWriter(log_dir=log_dir)

        # Prefill the replay buffer
        if self.initial_replay_buffer:
            self._prefill_buffer()

    def train(
            self,
            log_model_histogram=False,
            log_fidelity_histogram=True,
            adaptive_target_update=False,
            ):
        """
        Trains the agent in the given environment for the specified number of
        episodes.

        Parameters
        ----------
        log_model_histogram : bool, optional
            Whether to log histograms for model weights and biases. Defaults to
            False.

        Returns
        -------
        tuple
            A tuple containing the reward history, fidelity history, state
            history, omega history, delta history, amplitude history, phase
            history, and duration history.
        """
        for e in trange(self.episodes, desc="Training Episodes"):

            # Run episode
            total_reward = self._run_episode(e)

            # Train the agent
            loss = self.agent.replay()

            # Log loss if available
            if loss is not None:
                self.writer.add_scalar("Loss", loss, e)

            # Log learning rate, delta, epsilon
            current_lr = self.agent.lr_scheduler.get_last_lr()[0]
            current_delta = self.agent.delta
            current_epsilon = self.agent.epsilon
            self.writer.add_scalar("Learning Rate", current_lr, e)
            self.writer.add_scalar("Delta", current_delta, e)
            self.writer.add_scalar("Epsilon", current_epsilon, e)

            # Log histograms for model weights and biases if requested
            if log_model_histogram:
                self._log_model_histograms(e)

            # Log histograms for fidelities if requested
            if log_fidelity_histogram:
                self.writer.add_histogram("Fidelities", np.array(
                    self.metrics["fidelity"]), e)

            # Update target model periodically
            if adaptive_target_update:
                if self._adaptive_target_update(e):
                    self.agent.update_target_model()
            else:
                if e % self.target_update == 0 and e > 0:
                    self.agent.update_target_model()

            # Check for early stopping
            if self._check_early_stopping(total_reward, e):
                break

            # Free GPU memory
            empty_cache()

        print("\nTraining finished.")
        # Close the TensorBoard writer
        self.writer.close()

        return (
            self.metrics['total_reward'],
            self.metrics['fidelity'],
        )

    def _run_episode(self, episode):
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
            # Get action
            action = self.agent.act(state)
            # Step environment
            (
                done,
                next_state,
                discounted_reward,
                fidelity,
                log_infidelity,
            ) = self.env.step(action)

            # Remember experience
            self.agent.remember(
                state,
                action,
                discounted_reward,
                next_state,
                done
                )

            # Accumulate reward
            total_reward += discounted_reward

            # Update state
            state = next_state

            # Check if episode has ended
            if done:
                break

        # Store metrics
        self._store_episode_metrics(
            episode,
            state,
            discounted_reward,
            total_reward,
            fidelity,
            log_infidelity
        )
        # Log episode-specific metrics
        self._log_episode_metrics(
            episode,
            discounted_reward,
            total_reward,
            fidelity,
            )

        # Print progress
        if episode > 0 and episode % (self.episodes // 100) == 0:
            # Calculate fidelities for episode printing
            final_fidelity = self.metrics["fidelity"][-1]
            max_fidelity = np.max(self.metrics["fidelity"])
            min_fidelity = np.min(self.metrics["fidelity"])
            avg_fidelity = np.mean(self.metrics["fidelity"])
            median_fidelity = np.median(self.metrics["fidelity"])
            var_fidelity = np.var(self.metrics["fidelity"], ddof=1) if len(
                self.metrics["fidelity"]) > 1 else 0

            print(
                f"Episode: {episode}/{self.episodes}\n"
                f"Total Reward: {total_reward:.5f}\n"
                f"Final Fidelity: {final_fidelity:.6f}\n"
                f"Max. Fidelity: {max_fidelity:.6f}\n"
                f"Min. Fidelity: {min_fidelity:.6f}\n"
                f"Avg. Fidelity: {avg_fidelity:.6f}\n"
                f"Median Fidelity: {median_fidelity:.6f}\n"
                f"Fidelity Variance: {var_fidelity:.6f}\n"
                f"Current Epsilon: {self.agent.epsilon:.6f}\n"
                f"Current LR: {self.agent.lr_scheduler.get_last_lr()[0]}\n"
                f"Patience Counter: {self.patience_counter}\n"
            )
        return total_reward

    def _prefill_buffer(self):
        # Prefill the replay buffer
        """
        Prefills the replay buffer by generating random transitions until
        it reaches the specified minimum size. This is done at the start
        of training to ensure that the buffer is not empty before training
        begins.

        This function is usually called at the start of training,
        before the first episode is run. If the buffer is already filled,
        this function does nothing.
        """
        print("Prefilling the replay buffer...")
        while len(self.agent.memory) < config["hyperparameters"]["BATCH_SIZE"]:
            state = self.env.reset()
            for _ in range(self.env.max_steps):
                action = random.randrange(self.agent.action_size)
                done, next_state, discounted_reward, _, _,  = self.env.step(
                    action
                )

                self.agent.remember(
                    state,
                    action,
                    discounted_reward,
                    next_state,
                    done
                )

                state = next_state

                if done:
                    break

        print("\nReplay buffer prefilled. Start training...\n")

    def _check_early_stopping(self, total_reward, episode):
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

        trigger_reason = ""
        current_log_infidelity, current_fidelity = (
            self.metrics["log_infidelity"][-1],
            self.metrics["fidelity"][-1]
        )

        if current_log_infidelity > self.best_log_infidelity:
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

        if self.best_log_infidelity >= self.infidelity_threshold \
                or self.patience_counter >= self.patience:
            # Determine the reason for early stopping
            if self.best_log_infidelity >= self.infidelity_threshold \
               and self.patience_counter >= self.patience:
                trigger_reason = "Both Infidelity Threshold reached and\
                    patience counter filled"
            elif self.best_log_infidelity >= self.infidelity_threshold:
                trigger_reason = "Infidelity threshold reached"
            elif self.patience_counter >= self.patience:
                trigger_reason = "Patience counter filled"
            print(
                f"\nEarly stopping triggered by {trigger_reason}.\n"
                f"Best Achieved Reward: {total_reward:.6f}\n"
                f"Best Achieved Fidelity: {self.best_fidelity:.6f}\n"
                f"Episode: {episode}, Patience: {self.patience_counter}\n"
                f"Final Epsilon: {self.agent.epsilon:.6f}\n"
                f"Final LR: {self.agent.lr_scheduler.get_last_lr()[0]:.6f}\n"
            )

            return True

        return False

    def _adaptive_target_update(self, episode):
        """
        Decides whether to update the target model based on recent performance.
        """
        if len(self.metrics["log_infidelity"]) >= 10:
            recent_avg = np.mean(self.metrics["log_infidelity"][-10:])
            overall_avg = np.mean(self.metrics["log_infidelity"])
            return recent_avg < overall_avg
        return False

    def _store_episode_metrics(
            self,
            episode,
            state,
            total_reward,
            discounted_reward,
            fidelity,
            log_infidelity
            ):
        """
        Stores metrics for a single episode in the metrics dictionary.
        """
        self.metrics["episode"].append(episode)
        self.metrics["state"].append(state)
        self.metrics["total_reward"].append(total_reward)
        self.metrics["discounted_reward"].append(discounted_reward)
        self.metrics["fidelity"].append(fidelity)
        self.metrics["log_infidelity"].append(log_infidelity)

    def _log_episode_metrics(
            self,
            episode,
            discounted_reward,
            total_reward,
            fidelity,

            ):
        """
        Logs metrics for a single episode to TensorBoard.
        """
        self.writer.add_scalar("Discounted Reward", discounted_reward, episode)
        self.writer.add_scalar("Total Reward", total_reward, episode)
        self.writer.add_scalar("Fidelity", fidelity, episode)

    def _log_model_histograms(self, episode):
        """
        Logs histograms of model weights and gradients to TensorBoard.

        Parameters:
        ----------
        episode : int
            The current episode number.
        """
        for name, param in self.agent.model.named_parameters():
            if param.requires_grad:
                # Log weights
                self.writer.add_histogram(
                    f"Weights/{name}", param.data.cpu().numpy(), episode
                )
                # Log gradients only if they exist
                if param.grad is not None:
                    self.writer.add_histogram(
                        f"Gradients/{name}", param.grad.cpu().numpy(), episode
                    )
