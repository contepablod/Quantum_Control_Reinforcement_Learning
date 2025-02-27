import argparse
import math
import os
import sys
import torch
import torch.nn as nn
import platform
import psutil
import subprocess
from hyperparameters import config
from torch.optim import Optimizer


class AdamWNGD(Optimizer):
    """
    Implements Natural Gradient Descent with weight decay and AMSGrad.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        eps=1e-8,
        weight_decay=0.01,
        fisher_update_steps=10,
        amsgrad=False,
        fisher_mode="diagonal",
    ):
        """
        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate.
            eps (float): A small constant added for numerical stability.
            weight_decay (float): L2 regularization factor (weight decay).
            fisher_update_steps (int): Number of steps between Fisher
            information matrix updates.
            amsgrad (bool): Whether to use the AMSGrad variant of the
            optimizer.
            fisher_mode (str): Type of Fisher matrix ('diagonal', 'block',
            or 'full').
        """
        defaults = dict(
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            fisher_update_steps=fisher_update_steps,
            amsgrad=amsgrad,
            fisher_mode=fisher_mode,
        )
        if fisher_mode not in ["diagonal", "block", "full"]:
            raise ValueError(
                "Invalid fisher_mode. Choose from 'diagonal', 'block', \
                    or 'full'."
            )

        super(AdamWNGD, self).__init__(params, defaults)
        self.fisher_information = {}

    def _update_fisher_information(self, params, fisher_mode):
        """
        Updates the Fisher information matrix based on the selected mode.
        """
        for p in params:
            if p.grad is None:
                continue

            grad = p.grad.data
            if p not in self.fisher_information:
                if fisher_mode == 'full':
                    self.fisher_information[p] = torch.zeros_like(
                        grad.unsqueeze(-1).matmul(grad.unsqueeze(0))
                        )
                else:
                    self.fisher_information[p] = torch.zeros_like(grad)

            if fisher_mode == 'diagonal':
                # Approximate Fisher matrix with its diagonal
                self.fisher_information[p] = self.fisher_information[p] \
                    * 0.9 + grad**2 * 0.1
            elif fisher_mode == 'block':
                # Approximate Fisher with block structure (e.g., split grad
                # into blocks)
                block_size = max(1, grad.numel() // 4)  # Adjust block size
                # as needed
                grad_blocks = grad.split(block_size)
                fisher_blocks = self.fisher_information[p].split(block_size)
                self.fisher_information[p] = torch.cat([
                    fb * 0.9 + gb**2 * 0.1 for fb, gb in zip(
                        fisher_blocks,
                        grad_blocks
                        )
                ])
            elif fisher_mode == 'full':
                # Full Fisher matrix
                fisher_grad = grad.unsqueeze(-1).matmul(grad.unsqueeze(0))
                self.fisher_information[p] = self.fisher_information[p] * 0.9 \
                    + fisher_grad * 0.1

    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            fisher_update_steps = group["fisher_update_steps"]
            amsgrad = group["amsgrad"]
            fisher_mode = group["fisher_mode"]

            # Update Fisher information every `fisher_update_steps` iterations
            if "step_count" not in group:
                group["step_count"] = 0
            group["step_count"] += 1

            if group["step_count"] % fisher_update_steps == 0:
                self._update_fisher_information(group["params"], fisher_mode)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "NaturalGradientDescent does not support\
                        sparse gradients."
                    )

                # Apply weight decay
                if group["weight_decay"] != 0:
                    grad.add_(p.data, alpha=group["weight_decay"])

                # Use Fisher information matrix approximation to scale
                # gradients
                fisher_diag = self.fisher_information.get(p, torch.ones_like(
                    grad)
                    )
                if fisher_mode == "full":
                    fisher_diag = fisher_diag.diagonal()
                adjusted_grad = grad / (fisher_diag.sqrt() + group["eps"])

                # AMSGrad: Maintain the maximum of the Fisher information
                if amsgrad:
                    if "max_fisher" not in self.state[p]:
                        self.state[p]["max_fisher"] = torch.zeros_like(
                            fisher_diag
                            )
                    max_fisher = self.state[p]["max_fisher"]
                    torch.maximum(max_fisher, fisher_diag, out=max_fisher)
                    fisher_diag = max_fisher

                # Parameter update
                p.data.add_(adjusted_grad, alpha=-group["lr"])

        return loss


class DynamicHuberLoss(nn.Module):
    def __init__(self, delta_initial, delta_min, delta_decay_rate):
        """
        Initialize the Dynamic Huber Loss module.

        Parameters:
        ----------
        delta_initial : float
            The initial value of delta.
        delta_min : float
            The minimum value of delta (final value).
        total_steps : int
            The total number of steps for decay.
        """
        super(DynamicHuberLoss, self).__init__()
        self.delta_initial = delta_initial
        self.delta_min = delta_min
        self.delta_decay_rate = delta_decay_rate

    def forward(self, input, target, episode):
        """
        Compute the Huber loss with dynamically adjusted delta.

        Parameters:
        ----------
        input : torch.Tensor
            The predicted values.
        target : torch.Tensor
            The ground truth values.

        Returns:
        -------
        loss : torch.Tensor
            The computed Huber loss.
        """
        # Dynamically adjust delta
        delta = self.get_delta(episode)
        # Huber loss formula
        diff = torch.abs(input - target)
        quadratic = torch.min(diff, torch.tensor(delta, device=diff.device))
        linear = diff - quadratic
        loss = 0.5 * quadratic**2 + delta * linear
        return loss.mean(), delta

    def get_delta(self, episode):
        """
        Compute the current delta based on the step.

        Returns:
        -------
        delta : float
            The current value of delta.
        """
        self.delta = self.delta_min + (
            (self.delta_initial - self.delta_min)
            * math.exp(-self.delta_decay_rate * episode)
        )
        return self.delta


def parse_experiment_arguments():
    """
    Parses and validates command-line arguments for the quantum control
    experiment.

    Returns:
    --------
    argparse.Namespace
        A namespace containing the parsed and validated arguments.
    """
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Quantum Control Experiment Parameters"
    )

    # Define optional arguments with default values
    parser.add_argument(
        "--gate",
        type=str,
        default="H",
        help="Gate type (e.g., H, T or CNOT) [default: H]",
    )
    parser.add_argument(
        "--hamiltonian_type",
        type=str,
        default="Field",
        help="Hamiltonian type (e.g., Field, Rotational, etc.)\
            [default: Field]",
    )
    parser.add_argument(
        "--control_pulse_type",
        type=str,
        default="Discrete",
        help="Control pulse type (e.g., Discrete, Continuous, etc.)\
            [default: Discrete]",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="DDDQN",
        help="Agent type (e.g., DDDQN, TD3) [default: DDDQN]",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Validate each argument
    if args.gate not in ["H", "T", "CNOT"]:
        raise ValueError("Invalid gate type. Choose 'H', 'T', or 'CNOT'.")

    if args.hamiltonian_type not in ["Field", "Field_with_phase" ,"Rotational", "Geometric",
                                     "Composite"]:
        raise ValueError(
            "Invalid Hamiltonian type. Choose 'Field', 'Geometric'"
        )

    if args.control_pulse_type not in ["Discrete", "Continuous"]:
        raise ValueError(
            "Invalid control pulse type. Choose 'Discrete' or 'Continuous'."
        )

    if args.agent_type not in ["TD3", "DDDQN", "PPO", "GP"]:        
        raise ValueError(
            "Invalid agent type. Choose 'TD3', 'PPO', or 'DDDQN'."
        )

    return args


def gpu_management():
    """
    Configure the system for efficient GPU and CPU resource utilization
    """
    _kill_python_gpu_processes()
    # Thread and CPU Management
    os.environ["OMP_NUM_THREADS"] = str(
        os.cpu_count()
    )  # Use all available physical cores
    os.environ["OMP_PROC_BIND"] = "CLOSE"  # Bind threads close to the process
    os.environ["OMP_SCHEDULE"] = "DYNAMIC"  # Dynamic load balancing
    # for threads
    os.environ["KMP_BLOCKTIME"] = (
        "1"  # Quick thread turnover for RL or high-throughput GPU workloads
    )
    os.environ["KMP_AFFINITY"] = (
        "granularity=fine,scatter"  # Spread threads across CPUs for
        # NUMA systems
    )

    # GPU Settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:128"  # Set CUDA memory split size
    )
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Retain async execution
    # for performance

    # PyTorch Backend Configurations
    torch.backends.cudnn.enabled = True  # Enable cuDNN
    torch.backends.cudnn.allow_tf32 = True  # Allow TensorFloat32 for speed
    torch.backends.cuda.matmul.allow_tf32 = (
        True  # Allow TensorFloat32 for matrix multiplications
    )
    torch.backends.cudnn.benchmark = (
        True  # Disable cuDNN auto-tuner; safe for variable input sizes
    )
    torch.backends.cudnn.deterministic = False  # Disable cuDNN deterministic
    torch.cuda.empty_cache()  # Free unused GPU memory
    print("=" * 100)
    print("🛠️  Configured GPU!")
    _gpu_info()


def _gpu_info():
    """
    Print detailed information about the GPU, CPU, CUDA,
    and system configuration.
    """
    print("=" * 100)
    print("🛠️  System Configuration\n")

    # Python and PyTorch versions
    print(f"🔧 Python VERSION: {sys.version}")
    print(f"🔧 PyTorch VERSION: {torch.__version__}")

    # CUDA and cuDNN versions
    if torch.cuda.is_available():
        print("\n🔧 CUDA VERSION:")
        os.system("nvidia-smi")  # Display NVIDIA GPU stats
        print(f"\n🔧 cuDNN VERSION: {torch.backends.cudnn.version()}")
        os.system("nvcc --version")  # Display NVIDIA compiler version

        # CUDA Devices
        print("\n🔧 Number of CUDA Devices:", torch.cuda.device_count())
        print("🔧 Active CUDA Device: GPU", torch.cuda.current_device())
        print("🔧 GPU Name:", torch.cuda.get_device_name(
            torch.cuda.current_device())
            )
        print("🔧 GPU Memory Allocated:", torch.cuda.memory_allocated(),
              "bytes"
              )
        print("🔧 GPU Memory Reserved:", torch.cuda.memory_reserved(),
              "bytes")
    else:
        print("⚠️ CUDA is not available! Switching to CPU mode.")

    # CPU Information
    print("\n🔧 CPU Specifications:")
    print(f"🔧 Processor: {platform.processor()}")
    print(f"🔧 Architecture: {platform.architecture()[0]}")
    print(f"🔧 Machine: {platform.machine()}")
    print(f"🔧 System: {platform.system()} {platform.release()}")
    print(f"🔧 CPU Cores (Logical): {psutil.cpu_count(logical=True)}")
    print(f"🔧 CPU Cores (Physical): {psutil.cpu_count(logical=False)}")
    print(f"🔧 CPU Frequency: {psutil.cpu_freq().max:.2f} MHz")
    print(f"🔧 Total RAM: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")

    print("=" * 100)


def _kill_python_gpu_processes():
    """
    Identifies and kills all Python processes running on the GPU.
    """
    try:
        # Use nvidia-smi to get a list of processes on the GPU
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            print("Failed to query GPU processes:", result.stderr)
            return

        # Parse the output to find Python processes
        gpu_processes = result.stdout.strip().split("\n")
        for process in gpu_processes:
            if not process.strip():  # Skip empty lines
                continue

            try:
                pid, name = process.split(", ")
                if "python" in name.lower():
                    print(f"Terminating Python GPU process: PID {pid},\
                         Name {name}")
                    os.kill(int(pid), 9)  # SIGKILL signal to terminate process
            except ValueError as ve:
                print(f"Skipping malformed process line: {process}. Error:\
                      {ve}")

    except Exception as e:
        print(f"Error while terminating GPU processes: {e}")


def print_hyperparameters():
    print("🛠️  Training Hyperparameters")
    print("=" * 100)
    print(f"🔧 Number of Episodes: {config['hyperparameters']["train"]
                                   ["EPISODES"]}")
    print(f"🔧 Patience Limit: {config['hyperparameters']["train"]
                               ['PATIENCE']}")
    print(f"🔧 Latent Space: {config['hyperparameters']["general"]
                             ['HIDDEN_FEATURES']}")
    print(
        f"🔧 Scheduler Initial LR: {config['hyperparameters']["optimizer"]
                                   ['SCHEDULER_LEARNING_RATE']}"
    )
    print(
        f"🔧 Scheduler Type: {config['hyperparameters']["optimizer"]
                             ['SCHEDULER_TYPE']}"
    )
    if config["hyperparameters"]["optimizer"]["SCHEDULER_TYPE"] == "EXP":
        print(
            f"🔧 Scheduler Exp Decay: {config['hyperparameters']["optimizer"]
                                      ['EXP_LR_DECAY']}"
        )
        print(
        f"🔧 Scheduler Min LR: {config['hyperparameters']["optimizer"]
                               ['SCHEDULER_LR_MIN']}"
    )
    elif config["hyperparameters"]["optimizer"]["SCHEDULER_TYPE"] == "COS":
        print(
            f"🔧 Cosine WarmUp Steps: {config['hyperparameters']["optimizer"]
                                      ['COS_WARMUP_STEPS']}"
        )
        print(
            f"🔧 Cosine WarmUp Factor: {config['hyperparameters']["optimizer"]
                                       ['COS_WARMUP_FACTOR']}"
        )
        print(
            f"🔧 Cosine Max Period: {config['hyperparameters']["optimizer"]
                                    ['COS_T_MAX']}"
        )
    print(
        f"🔧 Optimizer: {config['hyperparameters']["optimizer"]
                        ['OPTIMIZER_TYPE']}"
    )
    print(f"🔧 Weight Decay: {config['hyperparameters']["optimizer"]
                             ['WEIGHT_DECAY']}")
    print(f"🔧 Loss Type: {config['hyperparameters']["loss"]['LOSS_TYPE']}")
    print(f"🔧 Discount Factor: {config['hyperparameters']["general"]
                                ['GAMMA']}")
    print(
        f"🔧 Fidelity Threshold: {config['hyperparameters']["train"]
                                 ['FIDELITY_THRESHOLD']}"
    )
    print("=" * 100)


def compile_model_torch(agent):
    # Compile the model (requires PyTorch 2.0 or later)
    """
    Compile the PyTorch model using torch.compile for faster inference.
    This is only supported in PyTorch 2.0 or later. If an earlier version
    is used, the model will not be compiled.

    Parameters:
    ----------
    agent : Pytorch model
        The agent-model to be compiled.

    Returns:
    -------
    agent : Pytorch model
        The agent-model compiled.
    """
    if torch.__version__ >= "2.0.0":
        print("🛠️  Compiling model...")
        print("=" * 100)
        agent.model = torch.compile(
            model=agent.model,
            fullgraph=True,
            dynamic=True,
            mode="max-autotune",
        )
        agent.target_model = torch.compile(
            model=agent.target_model,
            fullgraph=True,
            dynamic=True,
            mode="max-autotune",
        )


def get_execution_time(start_time, end_time):
    """
    Calculate the execution time of the training.
    """
    execution_time = end_time - start_time
    hours, rem = divmod(execution_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"Training completed in {int(hours)}h {int(minutes)}m\
    {int(seconds)}s"
