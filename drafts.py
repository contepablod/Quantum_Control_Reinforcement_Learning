import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity

# Example model and data
model = nn.Linear(100, 10).cuda()
data = torch.randn(1024, 100).cuda()
target = torch.randn(1024, 10).cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# Profiling
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
) as prof:
    with record_function("model_inference"):
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Print the profiling results
print(prof.key_averages().table(sort_by="cuda_time_total"))
print(torch.cuda.memory_summary(device=None, abbreviated=False))
