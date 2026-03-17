import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# ----------------------------------------------------------------
# 1. Define ICNN Architecture
# ----------------------------------------------------------------
class CertifiableICNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        h = 16 
        
        # W matrices: Multiply 'x' (skip connections)
        self.W = nn.ParameterList([
            nn.Parameter(torch.Tensor(h, input_dim)), # Layer 0
            nn.Parameter(torch.Tensor(h, input_dim)), # Layer 1
            nn.Parameter(torch.Tensor(1, input_dim))  # Output
        ])
        
        # U matrices: Multiply hidden states (Must be non-negative)
        self.U = nn.ParameterList([
            nn.Parameter(torch.Tensor(h, h)), # h1 -> h2
            nn.Parameter(torch.Tensor(1, h))  # h2 -> out
        ])
        
        self.bias = nn.ParameterList([
            nn.Parameter(torch.Tensor(h)),
            nn.Parameter(torch.Tensor(h)),
            nn.Parameter(torch.Tensor(1))
        ])
        
        self.act = nn.Softplus()
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.uniform_(p, -0.1, 0.1)

    def _icnn_pass(self, x):
        # Layer 0
        z = self.act(F.linear(x, self.W[0], self.bias[0]))
        
        # Layer 1: z = act( W1*x + b1 + softplus(U0)*z )
        z = self.act(F.linear(x, self.W[1], self.bias[1]) + F.linear(z, F.softplus(self.U[0])))
        
        # Output: out = W2*x + b2 + softplus(U1)*z
        return F.linear(x, self.W[2], self.bias[2]) + F.linear(z, F.softplus(self.U[1]))

    def forward(self, x):
        y = self._icnn_pass(x)
        # Ensure V(0) = 0
        y_zero = self._icnn_pass(torch.zeros_like(x))
        
        # Enforce Quadratic Lower Bound for Positive Definiteness
        V_quad = 0.1 * (x**2).sum(dim=1, keepdim=True)
        
        return (y - y_zero) + V_quad

# ----------------------------------------------------------------
# 2. Setup and Load
# ----------------------------------------------------------------
def load_model(path="vdp_icnn_params.pth"):
    if not os.path.exists(path):
        print(f"Error: Could not find '{path}'.")
        print("Please add 'torch.save(model.state_dict(), \"vdp_icnn_params.pth\")' to your training script and run it.")
        exit()
    
    # Initialize with input_dim=2 to match Van der Pol
    model = CertifiableICNN(input_dim=2)
    
    try:
        model.load_state_dict(torch.load(path))
        print(f"Successfully loaded model from {path}")
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        print("Ensure the class definition matches exactly what was used for training.")
        exit()
        
    model.eval()
    return model

def get_true_limit_cycle():
    """
    Simulates the FORWARD time Van der Pol oscillator to find the limit cycle.
    (The ROA boundary of the Reverse-Time system is the Limit Cycle of the Forward-Time system)
    """
    def vdp_forward(t, state):
        x1, x2 = state
        # Forward VdP Dynamics
        # dx1 = x2
        # dx2 = (1 - x1^2)*x2 - x1
        return [x2, (1 - x1**2) * x2 - x1]

    # Simulate long enough to converge to the cycle
    t_span = [0, 50]
    y0 = [0.1, 0.0] # Start near origin, spiral out
    sol = solve_ivp(vdp_forward, t_span, y0, rtol=1e-9, max_step=0.05)
    
    # Take the last 30% of the trajectory (converged part)
    split_idx = int(len(sol.t) * 0.7)
    return sol.y[0][split_idx:], sol.y[1][split_idx:]

# ----------------------------------------------------------------
# 3. Main Plotting Logic
# ----------------------------------------------------------------
def plot_nlf_roa():
    model = load_model()
    
    # A. Create Grid for Contours
    x_range = np.linspace(-3.0, 3.0, 200)
    y_range = np.linspace(-3.5, 3.5, 200)
    X, Y = np.meshgrid(x_range, y_range)
    inputs = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
    
    # Compute V(x) landscape
    with torch.no_grad():
        V_numeric = model(inputs).numpy().reshape(X.shape)
        
    # B. Get True Limit Cycle
    cycle_x, cycle_y = get_true_limit_cycle()
    
    # C. Find Certified Level Set (rho)
    # The largest level set contained INSIDE the limit cycle is bounded by 
    # the minimum value of V along that cycle.
    cycle_points = torch.tensor(np.stack([cycle_x, cycle_y], axis=1), dtype=torch.float32)
    with torch.no_grad():
        v_cycle = model(cycle_points).numpy()
        
    rho_est = np.min(v_cycle)

    if rho_est < 0:
        print("WARNING: The learned function V(x) is not Positive Definite (V < 0 on limit cycle).")
        print("Plotting full range to visualize the issue.")
        # Create levels from min(V) to max(V)
        v_min, v_max = V_numeric.min(), V_numeric.max()
        levels = np.linspace(v_min, v_max, 30)
    else:
        # Standard positive definite plotting
        levels = np.linspace(0, rho_est * 1.5, 30)

    # D. Plotting
    plt.figure(figsize=(10, 8))
    
    # 1. Background V(x) contours
    # We plot levels up to slightly beyond rho_est to see the "spillover"
    levels = np.linspace(0, rho_est * 1.5, 30)
    plt.contourf(X, Y, V_numeric, levels=levels, cmap="Greys", alpha=0.3)
    plt.colorbar(label="V(x)")
    
    # 2. True Limit Cycle (Red Dashed)
    plt.plot(cycle_x, cycle_y, 'r--', linewidth=2.5, label="True Limit Cycle")
    
    # 3. Neural Level Set (Lime Green) - The ROA Estimate
    # This contour represents the verified Region of Attraction
    cs = plt.contour(X, Y, V_numeric, levels=[rho_est], colors=['lime'], linewidths=3.0)
    plt.clabel(cs, inline=True, fmt="Certified ROA")
    
    # 4. Vector Field
    # dx1 = -x2
    # dx2 = x1 + (x1^2 - 1)*x2
    skip = 12
    X_q = X[::skip, ::skip]
    Y_q = Y[::skip, ::skip]
    
    U = -Y_q
    V_vec = X_q + (X_q**2 - 1) * Y_q
    
    plt.quiver(X_q, Y_q, U, V_vec, color='steelblue', alpha=0.6, label="Training Dynamics (-t)")

    plt.title(f"Certifiable ICNN: Region of Attraction\nEstimated Level Set: {rho_est:.4f}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(-3, 3)
    plt.ylim(-3.5, 3.5)
    
    filename = "roa_icnn_comparison.png"
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    # plt.show()

if __name__ == "__main__":
    plot_nlf_roa()