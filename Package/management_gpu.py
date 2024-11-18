import os
import torch
import sys

def gpu_management():
    os.environ["OMP_NUM_THREADS"] = "6"
    os.environ["OMP_PROC_BIND"] = "TRUE"
    os.environ["OMP_SCHEDULE"] = "STATIC"
    os.environ["GOMP_CPU_AFFINITY"] = "0-5"
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()

def gpu_info():
    print("__Python VERSION:", sys.version)
    print("\n__pyTorch VERSION:", torch.__version__)
    print("\n__CUDA VERSION")
    os.system("nvidia-smi")
    print("\n__CUDNN VERSION:", torch.backends.cudnn.version())
    os.system("nvcc --version")
    print("\n__Number CUDA Devices:", torch.cuda.device_count())
    print("\n__Devices")
    print("Available devices ", torch.cuda.device_count())
    print("Active CUDA Device: GPU", torch.cuda.current_device(), "\n")
