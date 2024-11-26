import os
import sys
import torch
import platform
import psutil


def gpu_management():
    """
    Configure the system for efficient GPU and CPU resource utilization
    """
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
        False  # Disable cuDNN auto-tuner; safe for variable input sizes
    )
    torch.cuda.empty_cache()  # Free unused GPU memory

    print("Configured GPU!")
    gpu_info()  # Call the GPU info function for detailed configuration display


def gpu_info():
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
        print("🔧 GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        print("🔧 GPU Memory Allocated:", torch.cuda.memory_allocated(), "bytes")
        print("🔧 GPU Memory Reserved:", torch.cuda.memory_reserved(), "bytes")
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
        agent.model = torch.compile(agent.model)
        agent.target_model = torch.compile(agent.target_model)
    return agent
