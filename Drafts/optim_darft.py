# import math
# import torch
# from torch.optim import Optimizer


# class QNOGOptimizer(Optimizer):
#     """
#     Quantum Natural Gradient Optimizer (QNOG) for PyTorch.

#     This optimizer adjusts parameter updates using the Quantum Fisher Information Matrix (QFIM),
#     mimicking the behavior of AdamW but adapted to the geometry of quantum states.
#     """

#     def __init__(
#         self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
#     ):
#         """
#         Initializes the QNOG optimizer.

#         Parameters:
#         -----------
#         params : iterable
#             Iterable of parameters to optimize or dicts defining parameter groups.
#         lr : float
#             Learning rate (default: 1e-3).
#         betas : tuple
#             Coefficients used for computing running averages of gradient and QFIM (default: (0.9, 0.999)).
#         eps : float
#             Term added to the denominator to improve numerical stability (default: 1e-8).
#         weight_decay : float
#             Weight decay (L2 penalty) (default: 0.01).
#         """
#         defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
#         super(QNOGOptimizer, self).__init__(params, defaults)

#     def _compute_qfim(self, params):
#         """
#         Compute the Quantum Fisher Information Matrix (QFIM) for the given parameters.

#         This is a placeholder method and should be implemented based on the specific
#         quantum problem and circuit.

#         Parameters:
#         -----------
#         params : torch.Tensor
#             Parameters of the quantum circuit.

#         Returns:
#         --------
#         torch.Tensor
#             QFIM matrix for the given parameters.
#         """
#         # Example QFIM as an identity matrix (for demonstration purposes).
#         # Replace this with a proper QFIM computation.
#         num_params = params.shape[0]
#         qfim = torch.eye(num_params, device=params.device)
#         return qfim

#     @torch.no_grad()
#     def step(self, closure=None):
#         """
#         Perform a single optimization step.

#         Parameters:
#         -----------
#         closure : callable, optional
#             A closure that reevaluates the model and returns the loss.

#         Returns:
#         --------
#         loss : float or None
#             Loss evaluated after the step (if closure is provided).
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()

#         for group in self.param_groups:
#             for param in group["params"]:
#                 if param.grad is None:
#                     continue

#                 grad = param.grad
#                 state = self.state[param]

#                 # State initialization
#                 if len(state) == 0:
#                     state["step"] = 0
#                     state["exp_avg"] = torch.zeros_like(param)
#                     state["exp_avg_sq"] = torch.zeros_like(param)

#                 exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
#                 beta1, beta2 = group["betas"]

#                 # Update biased first and second moment estimates
#                 exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
#                 exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

#                 # Compute bias-corrected estimates
#                 bias_correction1 = 1 - beta1 ** (state["step"] + 1)
#                 bias_correction2 = 1 - beta2 ** (state["step"] + 1)
#                 denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
#                     group["eps"]
#                 )

#                 # Compute QFIM
#                 qfim = self._compute_qfim(param)
#                 qfim_inv = torch.linalg.inv(
#                     qfim + group["eps"] * torch.eye(qfim.size(0), device=qfim.device)
#                 )

#                 # Update parameters with QNOG adjustment
#                 step_size = group["lr"] / bias_correction1
#                 param_update = step_size * qfim_inv @ exp_avg / denom
#                 param.add_(param_update, alpha=-1)

#                 # Apply weight decay
#                 if group["weight_decay"] != 0:
#                     param.add_(param, alpha=-group["weight_decay"])

#                 state["step"] += 1

#         return loss


# # Example model parameters
# params = torch.nn.Parameter(torch.randn(10))

# # Define the QNOG optimizer
# optimizer = QNOGOptimizer([params], lr=1e-3, weight_decay=0.01)


# # Example loss function
# def loss_fn():
#     return torch.sum(params**2)  # Example loss

# scaler = torch.amp.GradScaler('cpu')

# # Training loop
# for step in range(100):
#     optimizer.zero_grad()

#     with torch.amp.autocast('cpu'):
#         loss = loss_fn()

#     # Scale the loss and backward pass
#     scaler.scale(loss).backward()

#     # Perform the optimizer step
#     scaler.step(optimizer)
#     scaler.update()
#     print(f"Step {step}: Loss = {loss.item()}")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class NaturalGradientDescentOptimizer:
    def __init__(self, model, lr=1e-3, damping=1e-2):
        """
        Initialize the Natural Gradient Descent optimizer.

        Args:
            model (torch.nn.Module): The model whose parameters will be optimized.
            lr (float): Learning rate for the natural gradient descent.
            damping (float): Damping factor to regularize the Fisher matrix.
        """
        self.model = model
        self.lr = lr
        self.damping = damping

    def compute_fisher_matrix(self, data, loss_fn):
        """
        Compute the Fisher Information Matrix.

        Args:
            data (torch.Tensor): Input data.
            loss_fn (callable): Loss function.

        Returns:
            Fisher matrix (torch.Tensor).
        """
        self.model.zero_grad()
        output = self.model(data)
        loss = loss_fn(output)
        loss.backward(create_graph=True)

        fisher_matrix = []
        for param in self.model.parameters():
            if param.grad is not None:
                grad = param.grad.view(-1)
                fisher_matrix.append(torch.outer(grad, grad))

        return torch.stack(fisher_matrix).sum(dim=0)

    def step(self, data, loss_fn):
        """
        Perform a single optimization step.

        Args:
            data (torch.Tensor): Input data.
            loss_fn (callable): Loss function.
        """
        fisher_matrix = self.compute_fisher_matrix(data, loss_fn)
        fisher_matrix += self.damping * torch.eye(fisher_matrix.size(0))  # Add damping

        natural_gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                grad = param.grad.view(-1)
                natural_grad = torch.linalg.solve(fisher_matrix, grad)
                natural_gradients.append(natural_grad)

        # Update the parameters
        idx = 0
        for param in self.model.parameters():
            if param.grad is not None:
                param.data -= self.lr * natural_gradients[idx].view_as(param)
                idx += 1


# Example usage
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


model = SimpleModel()
optimizer = NaturalGradientDescentOptimizer(model, lr=1e-3, damping=1e-2)
loss_fn = nn.CrossEntropyLoss()

# Dummy data
data = Variable(torch.randn(5, 10))
target = Variable(torch.randint(0, 2, (5,)))

# Perform an optimization step
output = model(data)
loss = loss_fn(output, target)
optimizer.step(data, lambda: loss)
