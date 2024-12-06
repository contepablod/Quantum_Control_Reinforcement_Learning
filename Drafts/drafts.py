import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

# Dummy model and optimizer
model = torch.nn.Linear(10, 2).cuda()
optimizer = AdamW(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)
scaler = GradScaler()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training loop
for episode in range(5):
    optimizer.zero_grad()
    x = torch.randn(32, 10).cuda()
    target = torch.randint(0, 2, (32,)).cuda()

    with autocast(device_type=device.type):
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, target)
        print(f"Episode {episode}: Loss = {loss.item()}")

    scaler.scale(loss).backward()
    scaler.step(optimizer)  # Optimizer step
    scaler.update()

    if episode > 0:
        scheduler.step()  # Scheduler step
        print(f"Episode {episode}: LR = {optimizer.param_groups[0]['lr']}")
