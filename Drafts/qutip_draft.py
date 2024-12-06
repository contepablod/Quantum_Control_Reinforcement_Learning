import matplotlib.pyplot as plt
import numpy as np
from qutip import basis, tensor, sigmax, sigmay, sigmaz, qeye, mesolve, expect

# Define common elements
I = qeye(2)  # Identity
X = sigmax()  # Pauli-X
Y = sigmay()  # Pauli-Y
Z = sigmaz()  # Pauli-Z
ket0 = basis(2, 0)
ket1 = basis(2, 1)

# Time evolution
tlist = np.linspace(0, 1, 100)  # Time list for all gates

# Initialize results storage
results = {}

# 1. T-Gate
T_Hamiltonian = (np.pi / 4) * Z
psi0_T = (ket0 + ket1).unit()  # Initial state |+>
result_T = mesolve(T_Hamiltonian, psi0_T, tlist, [], [])
expectations_T = [expect(Z, state) for state in result_T.states]  # Z expectation

# Store T-gate results
results["T"] = (tlist, expectations_T)

# 2. H-Gate
RZ_pi2 = (np.pi / 2) * Z
RX_pi = (np.pi / 2) * X

# Initial state for H-gate
psi0_H = (ket0 + ket1).unit()

# Sequential evolutions
psi1_H = mesolve(RZ_pi2, psi0_H, tlist, [], [])
psi2_H = mesolve(RX_pi, psi1_H.states[-1], tlist, [], [])
psi_final_H = mesolve(RZ_pi2, psi2_H.states[-1], tlist, [], [])
expectations_H = [expect(Z, state) for state in psi_final_H.states]  # Z expectation

# Store H-gate results
results["H"] = (tlist, expectations_H)

# 3. CNOT Gate
ZZ = tensor(Z, Z)
IX = tensor(I, X)
CNOT_Hamiltonian = (np.pi / 4) * ZZ + (np.pi / 2) * IX

# Initial state for CNOT: |+>|0>
psi0_CNOT = tensor((ket0 + ket1).unit(), ket0)
result_CNOT = mesolve(CNOT_Hamiltonian, psi0_CNOT, tlist, [], [])
expectations_CNOT = [
    expect(tensor(I, Z), state) for state in result_CNOT.states
]  # Target Z

# Store CNOT results
results["CNOT"] = (tlist, expectations_CNOT)

def plot_dynamics(tlist, expectations, gate_name):
    """
    Plots the dynamics of the gate showing Z expectation over time.

    Parameters:
    - tlist: List of time points.
    - expectations: Expectation values at each time point.
    - gate_name: Name of the quantum gate (e.g., T, H, CNOT).
    """
    plt.figure(figsize=(8, 6))
    plt.plot(tlist, expectations, label=f"{gate_name}-Gate", drawstyle="steps-mid")
    plt.xlabel("Time")
    plt.ylabel("Expectation Value of Z")
    plt.title(f"Dynamics of {gate_name}-Gate")
    plt.grid()
    plt.legend()
    plt.show()
plot_dynamics(tlist, expectations_T, "T")
plot_dynamics(tlist, expectations_H, "H")
plot_dynamics(tlist, expectations_CNOT, "CNOT")
# Sample data
x = [0, 1, 2, 3, 4]
y = [1, 3, 2, 4, 3]

# Plot with different step styles
plt.figure(figsize=(10, 6))

plt.plot(x, y, drawstyle="default", label="Default", marker="o")
plt.plot(x, y, drawstyle="steps-pre", label="Steps-Pre", marker="o")
plt.plot(x, y, drawstyle="steps-mid", label="Steps-Mid", marker="o")
plt.plot(x, y, drawstyle="steps-post", label="Steps-Post", marker="o")

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Step Styles in Matplotlib")
plt.legend()
plt.grid()
plt.show()
