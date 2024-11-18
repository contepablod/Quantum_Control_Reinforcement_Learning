import numpy as np
import torch

def compile_model_torch(agent):
    # Compile the model (requires PyTorch 2.0 or later)
    if torch.__version__ >= "2.0.0":
        agent.model = torch.compile(agent.model)
        agent.target_model = torch.compile(agent.target_model)
    return agent
