import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# ----------------------------------------------------------------
# 1. Define Model Architecture
# ----------------------------------------------------------------
class LyapunovFunction(nn.Module):
    def __init__(self):
        super(LyapunovFunction, self).__init__()
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 1, bias=False) 
        self.act = nn.Tanh()

    def forward(self, x):
        h1 = self.act(self.layer1(x))
        h2 = self.act(self.layer2(h1))
        out = self.layer3(h2)
        V_quad = 0.1 * (x**2).sum(dim=1, keepdim=True)
        return out*out + V_quad

# ----------------------------------------------------------------
# 2. Setup and Load
# ----------------------------------------------------------------
def load_model(path="vdp_lyapunov_params.pth"):
    if not os.path.exists(path):
        print(f"Error: Could not find '{path}'. Run train_vdp.py first.")
        exit()
    
    model = LyapunovFunction()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def get_true_limit_cycle():
    """
    Simulates the FORWARD time Van der Pol oscillator.
    The limit cycle of the forward system is the boundary of attraction 
    for the reverse system we trained on.
    """
    def vdp_forward(t, state):
        x1, x2 = state
        # Standard VdP: dx1 = x2, dx2 = (1 - x1^2)x2 - x1
        # This matches the inverse of our training system.
        return [x2, (1 - x1**2) * x2 - x1]

    # Integrate long enough to settle on the cycle
    t_span = [0, 50]
    y0 = [0.1, 0.0] # Start near origin, spiral out to cycle
    sol = solve_ivp(vdp_forward, t_span, y0, rtol=1e-9, max_step=0.05)
    
    # Take the last 30% of points (steady state limit cycle)
    split_idx = int(len(sol.t) * 0.7)
    return sol.y[0][split_idx:], sol.y[1][split_idx:]

# ----------------------------------------------------------------
# 3. Main Plotting Logic
# ----------------------------------------------------------------
def plot_nlf_roa():
    model = load_model()
    
    # A. Create Grid
    x_range = np.linspace(-3.0, 3.0, 200)
    y_range = np.linspace(-3.5, 3.5, 200)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Flatten for PyTorch
    inputs = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
    
    # Compute V(x)
    with torch.no_grad():
        V_numeric = model(inputs).numpy().reshape(X.shape)
        
    # B. Get True Limit Cycle
    cycle_x, cycle_y = get_true_limit_cycle()
    
    # C. Find "Certified" Level Set (rho)
    # We evaluate V(x) along the true limit cycle. 
    # The minimum value is the largest level set fully contained within the cycle.
    cycle_points = torch.tensor(np.stack([cycle_x, cycle_y], axis=1), dtype=torch.float32)
    with torch.no_grad():
        v_cycle = model(cycle_points).numpy()
        
    rho_est = np.min(v_cycle)
    print(f"Estimated ROA Level Set (rho): {rho_est:.4f}")

    # D. Plotting
    plt.figure(figsize=(10, 8))
    
    # 1. Background V(x) contours
    plt.contourf(X, Y, V_numeric, levels=50, cmap="Greys", alpha=0.3)
    
    # 2. True Limit Cycle (Red Dashed)
    plt.plot(cycle_x, cycle_y, 'r--', linewidth=2.5, label="True Limit Cycle")
    
    # 3. Neural Level Set (Lime Green)
    # Plot the specific level set V(x) == rho
    cs = plt.contour(X, Y, V_numeric, levels=[rho_est], colors=['lime'], linewidths=3.5)
    plt.clabel(cs, inline=True, fmt="Max ROA")
    
    # 4. Vector Field (Reverse Time - Stability)
    # Quiver needs to show the flow towards the origin
    skip = 10
    X_q = X[::skip, ::skip]
    Y_q = Y[::skip, ::skip]
    # Dynamics: dx1 = -x2, dx2 = x1 + (x1^2 - 1)x2
    U = -Y_q
    V_vec = X_q + (X_q**2 - 1) * Y_q
    
    plt.quiver(X_q, Y_q, U, V_vec, color='steelblue', alpha=0.5)

    plt.title(f"Neural Lyapunov Function ROA\nEstimated Level Set: {rho_est:.4f}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(-3, 3)
    plt.ylim(-3.5, 3.5)
    
    filename = "roa_neural_comparison.png"
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    # plt.show() 
    
if __name__ == "__main__":
    plot_nlf_roa()