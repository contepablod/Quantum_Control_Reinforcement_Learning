import json
import numpy as np
import os
import pandas as pd
import pickle
import signal
import sys
import torch
from hyperparameters import config
from torch.cuda import empty_cache
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import trange


class BaseGPTrainer:
    def __init__(
        self,
        agent,
        env,
        episodes,
        patience,
        device,
        log_dir="runs",
        save_final_model=True,
        model_dir="",
        save_metrics=True,
        metrics_dir="",
        metrics_format="csv",  # Format: 'csv', 'json', or 'pickle'
    ):
        # Initialize attributes
        self.agent = agent
        self.env = env
        self.episodes = episodes
        self.patience = patience
        self.save_final_model = save_final_model
        self.model_dir = model_dir
        self.save_metrics = save_metrics
        self.metrics_dir = metrics_dir
        self.metrics_format = metrics_format
        self.device = device

        # Initialize a metrics dictionary
        self.metrics = {
            "episode": [],
            "total_reward": [],
            "fidelity": [],
            "log_infidelity": [],
            "avg_fidelity": [],
            "control_pulse_params": [],
            "time_step": [],
        }

        self.trajectories = {}

        # Early stopping variables
        self.best_log_infidelity = 0
        self.best_fidelity = 0
        self.best_avg_fidelity = 0
        self.patience_counter = 0

        # TensorBoard Writer
        self.writer = SummaryWriter(log_dir=log_dir)

    def _check_early_stopping(self, episode):
        trigger_reason = ""
        window_size = config["hyperparameters"]["train"]["WINDOW_SIZE"]

        current_log_infidelity, current_fidelity, current_avg_fidelity = (
            (
                np.mean(self.metrics["log_infidelity"][-window_size:])
                if len(self.metrics["log_infidelity"]) > window_size
                else 0
            ),
            (
                np.mean(self.metrics["fidelity"][-window_size:])
                if len(self.metrics["fidelity"]) > window_size
                else 0
            ),
            (
                np.mean(self.metrics["avg_fidelity"][-window_size:])
                if len(self.metrics["avg_fidelity"]) > window_size
                else 0
            ),
        )

        if current_log_infidelity > self.best_log_infidelity:
            self.best_log_infidelity = current_log_infidelity
            self.best_fidelity = current_fidelity
            self.best_avg_fidelity = current_avg_fidelity
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Log best infidelity
        self.writer.add_scalar("Best Log Infidelity", self.best_log_infidelity, episode)
        self.writer.add_scalar("Patience Counter", self.patience_counter, episode)
        if self.patience_counter >= self.patience:
            trigger_reason = "Patience counter filled"
            print(
                f"\nEarly stopping triggered by {trigger_reason}.\n"
                f"Best Achieved Fidelity: {self.best_fidelity:.6f}\n"
                f"Best Achieved Avg. Fidelity: {self.best_avg_fidelity:.6f}\n"
                f"Episode: {episode}, Patience: {self.patience_counter}\n"
            )
            return True
        elif (
            np.mean(self.metrics["avg_fidelity"][-window_size:])
            > self.env.fidelity_threshold
        ):
            trigger_reason = "Fidelity threshold reached"
            print(
                f"\nEarly stopping triggered by {trigger_reason}.\n"
                f"Best Achieved Fidelity: {self.best_fidelity:.6f}\n"
                f"Best Achieved Avg. Fidelity: {self.best_avg_fidelity:.6f}\n"
                f"Episode: {episode}, Patience: {self.patience_counter}\n"
            )
            return True

        return False

    def _store_episode_metrics(
        self,
        episode,
        total_reward,
        fidelity,
        log_infidelity,
        avg_fidelity,
        pulse_params,
        time_step,
    ):
        self.metrics["episode"].append(episode)
        self.metrics["total_reward"].append(total_reward)
        self.metrics["fidelity"].append(fidelity)
        self.metrics["log_infidelity"].append(log_infidelity)
        self.metrics["avg_fidelity"].append(avg_fidelity)
        self.metrics["control_pulse_params"].append(pulse_params)
        self.metrics["time_step"].append(time_step)

    def _log_episode_metrics(
        self,
        episode,
        total_reward,
        fidelity,
    ):
        self.writer.add_scalar("Total Reward", total_reward, episode)
        self.writer.add_scalar("Gate Fidelity", fidelity, episode)

    def _log_model_histograms(self, episode):
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

    def _save_final_model(self):
        final_model_path = os.path.join(
            self.model_dir,
            f"final_model_gate_{self.env.gate}_hamiltonian_{
                self.env.hamiltonian.hamiltonian_type}_agent_{
                    self.agent.agent_type}_pulse_{
                        self.env.pulse.control_pulse_type}.pt",
        )
        torch.save(self.agent.grpo.state_dict(), final_model_path)
        print(f"Final model saved: {final_model_path}\n")

    def _save_metrics(self):
        detached_metrics = {}
        for key, value in self.metrics.items():
            # Check if the metric is a list
            if isinstance(value, list):
                # Convert all tensors in the list to NumPy arrays
                detached_metrics[key] = [
                    v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                    for v in value
                ]
            else:
                # Handle non-list items (e.g., scalars)
                detached_metrics[key] = (
                    value.detach().cpu().numpy()
                    if isinstance(value, torch.Tensor)
                    else value
                )
        file_path = os.path.join(
            self.metrics_dir,
            f"training_metrics_gate_{self.env.gate}_hamiltonian_{
                self.env.hamiltonian.hamiltonian_type}_agent_{
                    self.agent.agent_type}_pulse_{
                        self.env.pulse.control_pulse_type}.{
                            self.metrics_format}",
        )
        if self.metrics_format == "csv":
            df = pd.DataFrame(detached_metrics)
            df.to_csv(file_path, index=False)
        elif self.metrics_format == "json":
            with open(file_path, "w") as f:
                json.dump(detached_metrics, f, indent=4)
        elif self.metrics_format == "pickle":
            with open(file_path, "wb") as f:
                pickle.dump(detached_metrics, f)
        else:
            raise ValueError(f"Unsupported metrics format: {self.metrics_format}")
        print(f"Metrics saved to: {file_path}\n")

    def _save_trajectory(self):
        trajectories = self.trajectories
        file_path = os.path.join(
            self.metrics_dir,
            f"trajectories_gate_{self.env.gate}_hamiltonian_{
                self.env.hamiltonian.hamiltonian_type}_agent_{
                    self.agent.agent_type}_pulse_{
                        self.env.pulse.control_pulse_type}.{
                            self.metrics_format}",
        )
        if self.metrics_format == "csv":
            df = pd.DataFrame(trajectories)
            df.to_csv(file_path, index=False)
        elif self.metrics_format == "json":
            with open(file_path, "w") as f:
                json.dump(trajectories, f, indent=4)
        elif self.metrics_format == "pickle":
            with open(file_path, "wb") as f:
                pickle.dump(trajectories, f)
        else:
            raise ValueError(
                f"Unsupported metrics format: {
                    self.metrics_format
                }"
            )
        print(f"Trajectories saved to: {file_path}\n")

    def _save_last_update(self):
        # Detach tensors in episode_data
        detached_metrics = {}
        for key, value in self.env.episode_data.items():
            # Check if the metric is a list
            if isinstance(value, list):
                # Convert all tensors in the list to NumPy arrays
                detached_metrics[key] = [
                    v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                    for v in value
                ]
            else:
                # Handle non-list items (e.g., scalars)
                detached_metrics[key] = (
                    value.detach().cpu().numpy()
                    if isinstance(value, torch.Tensor)
                    else value
                )

        file_path = os.path.join(
            self.metrics_dir,
            f"last_env_update_gate_{self.env.gate}_hamiltonian_{
                self.env.hamiltonian.hamiltonian_type}_agent_{
                    self.agent.agent_type}_pulse_{
                        self.env.pulse.control_pulse_type}.{
                            self.metrics_format}",
        )

        if self.metrics_format == "csv":
            df = pd.DataFrame(detached_metrics)
            df.to_csv(file_path, index=False)
        elif self.metrics_format == "json":
            with open(file_path, "w") as f:
                json.dump(detached_metrics, f, indent=4)
        elif self.metrics_format == "pickle":
            with open(file_path, "wb") as f:
                pickle.dump(detached_metrics, f)
        else:
            raise ValueError(f"Unsupported data format: {self.metrics_format}")

        print(f"Last Env Update saved to: {file_path}\n")


class GPTrainer(BaseGPTrainer):
    def __init__(
        self,
        agent,
        env,
        episodes,
        patience,
        device,
        log_dir="runs",
        save_final_model=True,
        model_dir="",
        save_metrics=True,
        metrics_dir="",
        metrics_format="csv",
    ):
        super().__init__(
            agent=agent,
            env=env,
            episodes=episodes,
            patience=patience,
            log_dir=log_dir,
            save_final_model=save_final_model,
            model_dir=model_dir,
            save_metrics=save_metrics,
            metrics_dir=metrics_dir,
            metrics_format=metrics_format,
            device=device,
        )

        self.timesteps = config["hyperparameters"]["PPO"]["TIMESTEPS"]
        self.epochs = config["hyperparameters"]["PPO"]["EPOCHS_PPO"]
        self.mini_batch_size = config["hyperparameters"]["general"]["BATCH_SIZE"]
        self.gamma = config["hyperparameters"]["general"]["GAMMA"]
        self.lam = config["hyperparameters"]["PPO"]["LAMBDA"]

    def train(
        self,
        log_model_histogram=False,
        log_fidelity_histogram=False,
    ):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTSTP, self._signal_handler)
        print("\nTraining started...\n")

        for e in trange(self.episodes, desc="Training Episodes"):

            # Run episode and collect total reward and trajectory
            trajectory = self._collect_trajectory(e)

            # # Learn from the trajectory
            total_loss, pol_loss, ent_loss = self.agent.update(trajectory, self.epochs)

            # Log loss if not None and not inf
            if total_loss is not None and not np.isinf(total_loss):
                self.writer.add_scalar("Loss", total_loss, e)
                self.writer.add_scalar("Policy Loss", pol_loss, e)
                self.writer.add_scalar("Entropy Loss", ent_loss, e)
                # self.writer.add_scalar("KL Loss", kl_loss, e)
            else:
                print("Loss is None or inf")

            # Log histograms for model weights and biases if requested
            if log_model_histogram:
                self._log_model_histograms(e)

            # Log histograms for fidelities if requested
            if log_fidelity_histogram:
                self.writer.add_histogram(
                    "Fidelities/Histogram", np.array(self.metrics["fidelity"]), e
                )
                self.writer.add_histogram(
                    "Advantages/Histogram", trajectory["advantages"], e
                )

            # Store metrics
            self._store_episode_metrics(
                e,
                self.total_reward,
                self.env.fidelity_gate,
                self.env.log_infidelity,
                self.env.avg_gate_fidelity,
                self.env.episode_data['control_pulse_params'],
                self.env.time_step,
            )

            # Log episode-specific metrics
            self._log_episode_metrics(
                e,
                self.total_reward,
                self.env.fidelity_gate,
            )

            # Print progress
            if e > 0 and e % (self.episodes // 100) == 0:
                print(
                    f"Episode: {e}/{self.episodes}\n"
                    f"Last Total Reward: {self.total_reward:.5f}\n"
                    f"Last Fidelity: {self.metrics["fidelity"][-1]:.6f}\n"
                    f"Last Max. Fidelity: {np.max(self.metrics["fidelity"]):.6f}\n"
                    f"Last Min. Fidelity: {np.min(self.metrics["fidelity"]):.6f}\n"
                    f"Last Mean Fidelity: {np.mean(
                        self.metrics["fidelity"]
                        ):.6f}\n"
                    f"Last Avg. Fidelity: {self.metrics['avg_fidelity'][-1]:.6f}\n"
                    f"Last Median Fidelity: {np.median(
                        self.metrics["fidelity"]
                        ):.6f}\n"
                    f"Last Fidelity Variance: {np.var(
                        self.metrics["fidelity"], ddof=1
                        ) if len(
                        self.metrics["fidelity"]
                        ) > 1 else 0:.6f}\n"
                    f"Last Fidelity Std. Dev.: {np.std(
                        self.metrics['fidelity'], ddof=1) if len(
                        self.metrics['fidelity']) > 1 else 0:.6f}\n"
                    f"Patience Counter: {self.patience_counter}\n"
                )

            # Check for early stopping
            if self._check_early_stopping(e):
                break

            # Free GPU memory
            empty_cache()

        print("\nTraining finished. Saving model...\n")

        # Save the final model
        if self.save_final_model:
            self._save_final_model()

        # Save metrics
        if self.save_metrics:
            self._save_metrics()

        # Save last update
        # self._save_last_update()
        self._save_trajectory()

        # Close TensorBoard writer
        self.writer.close()

    def _collect_trajectory(self, ep):

        states, actions, rewards, dones, log_probs = ([] for _ in range(5))
        state = self.env.reset()
        self.total_reward = 0

        # Collect trajectory
        for t in range(self.timesteps):
            torch.compiler.cudagraph_mark_step_begin()

            # Agent acts in the environment
            if self.agent.control_pulse_type == "Discrete":
                action, log_prob = self.agent.act(state)
            else:
                raw_action, action, log_prob = self.agent.act(state)
            done, next_state, step_reward = self.env.step(action)

            # Store trajectory data
            states.append(state)
            (
                actions.append(action)
                if self.agent.control_pulse_type == "Discrete"
                else actions.append(raw_action)
            )
            rewards.append(step_reward)
            dones.append(int(done))
            log_probs.append(log_prob)

            state = next_state
            self.total_reward += step_reward

            if done:
                if t < self.timesteps - 1:
                    state = self.env.reset()
                # break

        self.trajectories[ep] = self.env.episode_data

        returns = self._compute_discounted_rewards(rewards, dones)
        advantages = self._compute_group_advantages(returns)

        # Convert to arrays and return
        trajectory = {
            "states": np.array(states),
            "actions": np.array(actions),
            "log_probs": np.array(log_probs),
            "dones": np.array(dones),
            "advantages": advantages,
        }

        return trajectory

    def _compute_discounted_rewards(self, rewards, dones):
        discounted_rewards = []
        cumulative = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                cumulative = 0  # Reset at episode boundary
            cumulative = reward + self.gamma * cumulative
            discounted_rewards.insert(0, cumulative)  # Insert at the beginning
        return np.array(discounted_rewards)

    def _compute_group_advantages(self, returns):
        group_advantages = returns - returns.mean()
        group_advantages = group_advantages / (returns.std() + 1e-8)
        return group_advantages

    def _signal_handler(self, _sig, _frame):
        print("\nInterrupt received. Saving model and metrics before exiting...")
        if self.save_final_model:
            self._save_final_model()
        if self.save_metrics:
            self._save_metrics()
        # self._save_last_update()
        self._save_trajectory()
        self.writer.close()
        sys.exit(0)  # or os._exit(0) if necessary
