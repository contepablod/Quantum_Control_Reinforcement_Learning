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
    torch.backends.cudnn.deterministic = False  # Disable cuDNN deterministic mode
    torch.cuda.empty_cache()  # Free unused GPU memory

    print("Configured GPU!")
    _gpu_info()


def _gpu_info():
    """
    Print detailed information about the GPU, CPU, CUDA,
    and system configuration.
    """
    print("=" * 100)
    print("🛠️  System Configuration")
    print("=" * 100)

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
            pid, name = process.split(", ")
            if "python" in name.lower():
                print(f"Terminating Python GPU process: PID {pid},\
                      Name {name}"
                      )
                os.kill(int(pid), 9)  # SIGKILL signal to terminate process

    except Exception as e:
        print(f"Error while terminating GPU processes: {e}")


def print_hyperparameters():
    print("🛠️  Training Hyperparameters")
    print("=" * 100)
    print(f"🔧 Number of Episodes: {config['hyperparameters']['EPISODES']}")
    print(f"🔧 Patience Limit: {config['hyperparameters']['PATIENCE']}")
    print(f"🔧 Latent Space: {config['hyperparameters']['HIDDEN_FEATURES']}")
    print(f"🔧 Batch Size: {config['hyperparameters']['BATCH_SIZE']}")
    print(f"🔧 Scheduler Initial LR: {config['hyperparameters']
                                     ['SCHEDULER_LEARNING_RATE']}")
    print(f"🔧 Scheduler Type: {config['hyperparameters']['SCHEDULER_TYPE']}")
    if config['hyperparameters']['SCHEDULER_TYPE'] == 'EXP':
        print(f"🔧 Scheduler Exp Decay: {config['hyperparameters']
                                        ['EXP_LR_DECAY']}")
    elif config['hyperparameters']['SCHEDULER_TYPE'] == 'COS':
        print(
            f"🔧 Cosine WarmUp Steps: {config['hyperparameters']
                                      ['COS_WARMUP_STEPS']}"
        )
        print(f"🔧 Cosine WarmUp Factor: {config['hyperparameters']
                                         ['COS_WARMUP_FACTOR']}")
        print(
            f"🔧 Cosine Max Period: {config['hyperparameters']['COS_T_MAX']}"
        )
    print(f"🔧 Scheduler Min LR: {config['hyperparameters']
                                 ['SCHEDULER_LR_MIN']}")
    print(f"🔧 Weight Decay: {config['hyperparameters']['WEIGHT_DECAY']}")
    print(f"🔧 Epsilon Max: {config['hyperparameters']['MAX_EPSILON']}")
    print(f"🔧 Epsilon Min: {config['hyperparameters']['MIN_EPSILON']}")
    print(f"🔧 Epsilon Decay: {config['hyperparameters']
                              ['EPSILON_DECAY_RATE']}")
    print(f"🔧 Discount Factor: {config['hyperparameters']['GAMMA']}")
    print(f"🔧 Loss Type: {config['hyperparameters']['LOSS_TYPE']}")
    if config['hyperparameters']['LOSS_TYPE'] == 'HUBER_DYNAMIC':
        print(f"🔧 Huber Delta Max: {config['hyperparameters']
                                    ['HUBER_DELTA_MAX']}")
        print(f"🔧 Huber Delta Min: {config['hyperparameters']
                                    ['HUBER_DELTA_MIN']}")
        print(f"🔧 Huber Delta Decay: {config['hyperparameters']
                                      ['HUBER_DELTA_DECAY_RATE']}")
    print(f"🔧 Discount Factor: {config['hyperparameters']['GAMMA']}")
    print(f"🔧 Replay Memory Size: {config['hyperparameters']['MEMORY_SIZE']}")
    print(f"🔧 Target Update Frequency: {config['hyperparameters']
                                        ['TARGET_UPDATE']}")
    print(f"🔧 Max Steps: {config['hyperparameters']['MAX_STEPS']}")
    print(f"🔧 Infidelity Threshold: {config['hyperparameters']
                                     ['INFIDELITY_THRESHOLD']}")
    print("=" * 100)


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
    # parser.add_argument(
    #     "--fidelity_type",
    #     type=str,
    #     default="state",
    #     help="Fidelity type (e.g., state, gate) [default: state]",
    # )
    # parser.add_argument(
    #     "--basis_type",
    #     type=str,
    #     default="Z",
    #     help="Basis type (e.g., Z, X, etc.) [default: Z]",
    # )
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
        help="Agent type (e.g., DQN, DDQN, DDPG, DDDQN) [default: DDDQN]",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Validate each argument
    if args.gate not in ["H", "T", "CNOT"]:
        raise ValueError("Invalid gate type. Choose 'H', 'T', or 'CNOT'.")

    # if args.fidelity_type.lower() not in ["state", "gate", "map"]:
    #     raise ValueError("Invalid fidelity type. Choose 'state', 'gate', or \
    #                      'map'.")

    # if args.basis_type not in ["Z", "X", "Y", "random"]:
    #     raise ValueError("Invalid basis type. Choose 'Z', 'X', 'Y', \
    #                      r 'random'.")

    if args.hamiltonian_type not in ["Field", "Rotational", "Geometric",
                                     "Composite"]:
        raise ValueError(
            "Invalid Hamiltonian type. Choose 'Field', 'Rotational', \
            'Geometric', or 'Composite'."
        )

    if args.control_pulse_type not in ["Discrete", "Continuous"]:
        raise ValueError(
            "Invalid control pulse type. Choose 'Discrete' or 'Continuous'."
        )

    if args.agent_type not in ["DQN", "DDQN", "DDPG", "DDDQN"]:
        raise ValueError(
            "Invalid agent type. Choose 'DQN', 'DDQN', 'DDPG', or 'DDDQN'."
        )

    return args


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
        print("=" * 100, '\n')
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
