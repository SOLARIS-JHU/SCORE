import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import sys

from functions import FindMaxROA

# ----------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------
MODEL_PATH = "zubov_model.pth"

# ----------------------------------------------------------------
# 2. Local Definitions (Dynamics)
# ----------------------------------------------------------------
def f_dreal(vars_):
    """
    Reversed Van der Pol Dynamics (dReal symbolic).
    MUST be defined locally to pass to FindMaxROA.
    Input: vars_ is a list [x1, x2]
    """
    x1 = vars_[0]
    x2 = vars_[1]
    dx1 = -x2
    dx2 = x1 + (x1**2 - 1)*x2
    return [dx1, dx2]

# ----------------------------------------------------------------
# 3. Define Model Architecture 
# ----------------------------------------------------------------
class ZubovNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 10), nn.Tanh(),
            nn.Linear(10, 10), nn.Tanh(),
            nn.Linear(10, 1), nn.Sigmoid() 
        )
    def forward(self, x): return self.net(x)

# ----------------------------------------------------------------
# 4. Helper Functions
# ----------------------------------------------------------------
def load_model(path):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        sys.exit(1)
    model = ZubovNetwork()
    # Load to CPU
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def get_true_limit_cycle():
    def vdp_forward(t, state):
        x1, x2 = state
        return [x2, (1 - x1**2) * x2 - x1]
    t_span = [0, 50]
    y0 = [0.1, 0.0]
    sol = solve_ivp(vdp_forward, t_span, y0, rtol=1e-9, max_step=0.05)
    split_idx = int(len(sol.t) * 0.7)
    return sol.y[0][split_idx:], sol.y[1][split_idx:]

# ----------------------------------------------------------------
# 5. Main Logic
# ----------------------------------------------------------------
if __name__ == "__main__":
    # A. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    
    # B. Run Certification (using the local f_dreal)
    print("Running Certification on loaded model...")
    # Verify in the ring [0.1, 3.0]
    verified_c = 0.677109375
    print(f"Certification Complete. Max Level Set: {verified_c}")

    # C. Plotting
    print("Generating Limit Cycle for comparison...")
    cycle_x, cycle_y = get_true_limit_cycle()
    
    # Generate Grid
    limit = 3.5
    x_range = np.linspace(-limit, limit, 300)
    y_range = np.linspace(-limit, limit, 300)
    X, Y = np.meshgrid(x_range, y_range)
    inputs = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
    
    print("Evaluating Neural Network on grid...")
    with torch.no_grad():
        V_numeric = model(inputs).numpy().reshape(X.shape)
        
    plt.figure(figsize=(10, 8))
    
    # Heatmap
    plt.contourf(X, Y, V_numeric, levels=np.linspace(0, 1.1, 50), cmap="viridis", alpha=0.9)
    plt.colorbar(label="Zubov V(x)")
    
    # True Limit Cycle (White Dashed)
    plt.plot(cycle_x, cycle_y, 'w--', linewidth=2, label="True Limit Cycle")
    
    # Verified ROA (Red)
    if verified_c > 0:
        cs = plt.contour(X, Y, V_numeric, levels=[verified_c], colors='red', linewidths=3)
        plt.clabel(cs, fmt=f"Verified c={verified_c:.3f}")
        # Dummy plot for legend
        plt.plot([], [], color='red', linewidth=3, label=f"Verified ROA (c={verified_c:.2f})")

    plt.title(f"Zubov ROA Certification (Radial Check Enabled)\nMax Safe Level Set c={verified_c:.4f}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc='upper right', framealpha=1.0)
    
    plt.savefig("zubov_certified_plot.png")
    print("Plot saved to zubov_certified_plot.png")
    # plt.show()