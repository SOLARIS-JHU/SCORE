import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.path import Path
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import sys
import torch.nn.functional as F
import os

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ==========================================
# 0. Configuration & Style
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
    "figure.titlesize": 18,
    "mathtext.fontset": "cm", 
    "lines.linewidth": 2.5,
    "figure.constrained_layout.use": False
})

# ==========================================
# 1. System & Models
# ==========================================

# --- A. EVT (Dict-Gram) Import Check ---
evt_path = os.path.abspath("EVT/van_der_pol")
if evt_path not in sys.path:
    sys.path.append(evt_path)

try:
    from EVT.commons import ODEGramMatrixLyapunov
    EVT_DICT_GRAM_AVAILABLE = True
except ImportError:
    EVT_DICT_GRAM_AVAILABLE = False
    print("[WARN] Could not import ODEGramMatrixLyapunov. EVT+Dict-Gram will be skipped.")

# --- B. Dynamics ---
def vdp_dynamics(t, state):
    x1, x2 = state
    mu = 1.0
    dx1 = x2
    dx2 = mu * (1 - x1**2) * x2 - x1
    return [dx1, dx2]

# --- C. Polynomial Lyapunov Function ---
def V_polynomial_eval(x1, x2):
    """
    Evaluates the polynomial Lyapunov function V(x).
    Scale factor: 0.3286861744740211
    """
    SCALE = 0.3286861744740211
    
    val = (2.8415e-14 + 1.1406e-07 * x1 - 1.7804e-07 * x2 
           - 0.5679 * (x1 * x2) + 0.0005 * (x1 * x2**2) 
           - 0.0269 * (x1 * x2**3) + 0.0014 * (x1 * x2**4) 
           + 3.0618e-05 * (x1 * x2**5) - 0.0138 * (x1**2 * x2) 
           - 0.0451 * (x1**2 * x2**2) + 0.0003 * (x1**2 * x2**3) 
           + 0.0013 * (x1**2 * x2**4) + 0.2220 * (x1**3 * x2) 
           - 0.0027 * (x1**3 * x2**2) + 0.0026 * (x1**3 * x2**3) 
           + 0.0033 * (x1**4 * x2) - 0.0058 * (x1**4 * x2**2) 
           - 0.0161 * (x1**5 * x2) + 1.0004 * x1**2 
           + 0.0195 * x1**3 - 0.2397 * x1**4 
           - 0.0060 * x1**5 + 0.0250 * x1**6 
           + 0.3822 * x2**2 - 0.0057 * x2**3 
           - 0.0011 * x2**4 - 5.5311e-06 * x2**5 
           + 3.3343e-06 * x2**6)
    
    return val / SCALE

# --- D. Neural Network Architectures ---

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

class ZubovNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 10), nn.Tanh(),
            nn.Linear(10, 10), nn.Tanh(),
            nn.Linear(10, 1), nn.Sigmoid() 
        )
    def forward(self, x):
        return self.net(x)

class ICNN(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        h = 16 
        
        self.W = nn.ParameterList([
            nn.Parameter(torch.Tensor(h, input_dim)),
            nn.Parameter(torch.Tensor(h, input_dim)),
            nn.Parameter(torch.Tensor(1, input_dim))
        ])
        
        self.U = nn.ParameterList([
            nn.Parameter(torch.Tensor(h, h)),
            nn.Parameter(torch.Tensor(1, h))
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
        z = self.act(F.linear(x, self.W[0], self.bias[0]))
        z = self.act(F.linear(x, self.W[1], self.bias[1]) + F.linear(z, F.softplus(self.U[0])))
        return F.linear(x, self.W[2], self.bias[2]) + F.linear(z, F.softplus(self.U[1]))

    def forward(self, x):
        y = self._icnn_pass(x)
        y_zero = self._icnn_pass(torch.zeros_like(x))
        V_quad = 0.1 * (x**2).sum(dim=1, keepdim=True)
        return (y - y_zero) + V_quad

# ==========================================
# 2. Helpers
# ==========================================
def count_pixels(mask):
    return np.sum(mask)

def calculate_coverage(method_mask, limit_cycle_mask):
    lc_pixels = count_pixels(limit_cycle_mask)
    if lc_pixels == 0: return 0.0
    intersection = method_mask & limit_cycle_mask
    return count_pixels(intersection) / lc_pixels

def calculate_area_ratio(method_mask, limit_cycle_mask):
    lc_pixels = count_pixels(limit_cycle_mask)
    if lc_pixels == 0: return 0.0
    method_pixels = count_pixels(method_mask)
    return method_pixels / lc_pixels

# ==========================================
# 3. Main Routine
# ==========================================
def main():
    device = torch.device("cpu")
    
    # 1. MODIFIED GRID SETUP
    grid_res = 400 
    x_range = np.linspace(-2.0, 2.0, grid_res)
    y_range = np.linspace(-3.5, 3.5, grid_res)
    
    X, Y = np.meshgrid(x_range, y_range)
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)

    print("Evaluating Models on Grid...")
    
    # --- Evaluation ---
    V_poly_grid = V_polynomial_eval(X, Y)

    nlf_model = LyapunovFunction().to(device)
    path_nlf = "NLF/vdp_lyapunov_params.pth"
    if os.path.exists(path_nlf):
        nlf_model.load_state_dict(torch.load(path_nlf, map_location=device))
        nlf_model.eval()
        with torch.no_grad():
            V_nlf_grid = nlf_model(grid_tensor).cpu().numpy().reshape(X.shape)
    else:
        print("NLF weights missing, skipping...")
        V_nlf_grid = np.zeros_like(X)

    icnn_model = ICNN().to(device)
    path_icnn = "ICNN/vdp_icnn_params.pth"
    has_icnn = False
    if os.path.exists(path_icnn):
        icnn_model.load_state_dict(torch.load(path_icnn, map_location=device))
        icnn_model.eval()
        with torch.no_grad():
            V_icnn_grid = icnn_model(grid_tensor).cpu().numpy().reshape(X.shape)
        has_icnn = True
    else:
        print("ICNN weights missing. Run training first.")
        V_icnn_grid = np.zeros_like(X)

    zubov_model = ZubovNetwork().to(device)
    path_zubov = "PINN/zubov_model.pth"
    if os.path.exists(path_zubov):
        zubov_model.load_state_dict(torch.load(path_zubov, map_location=device))
        zubov_model.eval()
        with torch.no_grad():
            V_zubov_grid = zubov_model(grid_tensor).cpu().numpy().reshape(X.shape)
    else:
        V_zubov_grid = np.zeros_like(X)

    V_evt_dict_gram_grid = None
    if EVT_DICT_GRAM_AVAILABLE:
        try:
            evt_model = ODEGramMatrixLyapunov(state_dim=2, feature_dim=10, device=device).to(device)
            path_evt = "EVT/models/lyapunov_model.pth"
            if os.path.exists(path_evt):
                evt_model.load_state_dict(torch.load(path_evt, map_location=device))
                evt_model.eval()
                with torch.no_grad():
                    out = evt_model(grid_tensor)
                    if isinstance(out, tuple): out = out[0]
                    V_evt_dict_gram_grid = out.cpu().numpy().reshape(X.shape)
            else:
                print("[WARN] EVT+Dict-Gram model weights not found.")
        except Exception as e:
            print(f"EVT Error: {e}")

    # LQR
    A = np.array([[0, 1], [-1, 1]]); B = np.array([[0], [1]])
    Q = np.eye(2); R = np.array([[1]])
    P = solve_continuous_are(A, B, Q, R)
    V_lqr_grid = P[0,0]*X**2 + (P[0,1] + P[1,0])*X*Y + P[1,1]*Y**2

    # --- Limit Cycle Simulation ---
    print("Simulating Limit Cycle...")
    sol = solve_ivp(vdp_dynamics, [0, 50], [2.0, 0.0], max_step=0.01, rtol=1e-9, atol=1e-9)
    x_stable, y_stable = sol.y[0, -1000:], sol.y[1, -1000:]
    verts = np.column_stack([x_stable, y_stable])
    verts = np.vstack([verts, verts[0]]) 
    lc_path = Path(verts)
    limit_cycle_mask = lc_path.contains_points(grid_points, radius=0.0).reshape(X.shape)

    rho_icnn = 0.0
    if has_icnn:
        cycle_tensor = torch.tensor(np.column_stack((x_stable, y_stable)), dtype=torch.float32).to(device)
        with torch.no_grad():
            v_cycle_vals = icnn_model(cycle_tensor).cpu().numpy()
        rho_icnn = np.min(v_cycle_vals)

    # --- Metrics ---
    rho_vals = {
        'SOS': 4.990614,        
        'SOS+EVT': 4.5454,      
        'NLF+SMT': 0.7237, 
        'Zubov+SMT': 0.6771, 
        'ICNN+SMT': rho_icnn,
        'Dict-Gram+EVT': 1.6277    
    }
    
    metrics = {}
    
    def get_metrics(name, V, rho):
        if V is None or rho == 0: return 0,0
        mask = V <= rho
        cov = calculate_coverage(mask, limit_cycle_mask)
        area = calculate_area_ratio(mask, limit_cycle_mask)
        return cov, area

    # Adjusted to match exactly the new keys
    metrics['SOS'] = get_metrics('SOS', V_poly_grid, rho_vals['SOS'])
    metrics['SOS+EVT'] = get_metrics('SOS+EVT', V_poly_grid, rho_vals['SOS+EVT'])
    metrics['NLF+SMT'] = get_metrics('NLF+SMT', V_nlf_grid, rho_vals['NLF+SMT'])
    metrics['Zubov+SMT'] = get_metrics('Zubov+SMT', V_zubov_grid, rho_vals['Zubov+SMT'])
    metrics['ICNN+SMT'] = get_metrics('ICNN+SMT', V_icnn_grid, rho_vals['ICNN+SMT'])

    if V_evt_dict_gram_grid is not None:
        metrics['Dict-Gram+EVT'] = get_metrics('Dict-Gram+EVT', V_evt_dict_gram_grid, rho_vals['Dict-Gram+EVT'])

    print("\n=== ROA Comparison ===")
    print(f"SOS (Pure)    : ρ={rho_vals['SOS']:.4f}, Cov={metrics['SOS'][0]*100:.1f}%")
    print(f"SOS+EVT       : ρ={rho_vals['SOS+EVT']:.4f}, Cov={metrics['SOS+EVT'][0]*100:.1f}%")

    # ==========================================
    # 4. PLOTTING
    # ==========================================
    print("\nGenerating Figure...")
    
    fig, ax = plt.subplots(figsize=(9, 9))
    plt.subplots_adjust(bottom=0.20) 

    # A. Vector Field (Time-Reversed for Stability Analysis)
    skip = 16 
    X_q, Y_q = X[::skip, ::skip], Y[::skip, ::skip]
    
    # REVERSED DYNAMICS (Stable Origin)
    U = -Y_q
    V = -((1 - X_q**2)*Y_q - X_q)
    
    M = np.hypot(U, V)
    M[M == 0] = 1.0
    ax.quiver(X_q, Y_q, U/M, V/M, pivot='mid', color='#666666', 
              scale=25, width=0.003, alpha=0.5, headwidth=4, headlength=5)

    # B. High Contrast Styles - UPDATED KEYS
    styles = {
        'SOS':           ('#377eb8', ':',  2.5),  # Blue
        'SOS+EVT':       ('#000080', '-',  3.5),  # Navy Blue
        'NLF+SMT':       ('#e41a1c', '--', 3.0),  # Bright Red
        'Zubov+SMT':     ('#4daf4a', '-.', 3.0),  # Green
        'ICNN+SMT':      ('#ff7f00', '-.', 3.0),  # Bright Orange
        'Dict-Gram+EVT': ('#984ea3', '--', 3.0),  # Purple
    }

    # C. Plot Contours
    ax.contourf(X, Y, V_poly_grid, levels=[0, rho_vals['SOS+EVT']], 
                colors=[styles['SOS+EVT'][0]], alpha=0.05)

    # Draw Lines - UPDATED KEYS
    if has_icnn:
        ax.contour(X, Y, V_icnn_grid, levels=[rho_vals['ICNN+SMT']], 
                   colors=styles['ICNN+SMT'][0], linestyles=styles['ICNN+SMT'][1], linewidths=styles['ICNN+SMT'][2])

    ax.contour(X, Y, V_nlf_grid, levels=[rho_vals['NLF+SMT']], 
               colors=styles['NLF+SMT'][0], linestyles=styles['NLF+SMT'][1], linewidths=styles['NLF+SMT'][2])
    
    ax.contour(X, Y, V_zubov_grid, levels=[rho_vals['Zubov+SMT']], 
               colors=styles['Zubov+SMT'][0], linestyles=styles['Zubov+SMT'][1], linewidths=styles['Zubov+SMT'][2])

    if V_evt_dict_gram_grid is not None:
        ax.contour(X, Y, V_evt_dict_gram_grid, levels=[rho_vals['Dict-Gram+EVT']], 
                   colors=styles['Dict-Gram+EVT'][0], linestyles=styles['Dict-Gram+EVT'][1], linewidths=styles['Dict-Gram+EVT'][2])

    ax.contour(X, Y, V_poly_grid, levels=[rho_vals['SOS']], 
               colors=styles['SOS'][0], linestyles=styles['SOS'][1], linewidths=styles['SOS'][2])

    ax.contour(X, Y, V_poly_grid, levels=[rho_vals['SOS+EVT']], 
               colors=styles['SOS+EVT'][0], linestyles=styles['SOS+EVT'][1], linewidths=styles['SOS+EVT'][2])

    # Limit Cycle (Black)
    ax.plot(x_stable, y_stable, color='black', linewidth=3.5, label='Limit Cycle', alpha=0.9)

    # D. Formatting
    ax.set_xlabel(r'$x_1$', labelpad=10, fontsize=18)
    ax.set_ylabel(r'$x_2$', labelpad=10, fontsize=18)
    
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-3.5, 3.5)
    
    ax.tick_params(direction='in', top=True, right=True, which='both', labelsize=14)
    ax.grid(False) 

    # E. Legend
    def fmt_legend(name, metrics_dict):
        c, r = metrics_dict.get(name, (0,0))
        # This will render full names with math-style bolding nicely in matplotlib
        return r"$\bf{" + name + r"}$" + r" ($\kappa$={0:.0f}%)".format(c*100)

    # Legend - UPDATED KEYS
    legend_elements = [
        Line2D([0], [0], color='black', lw=3.5, label='Limit Cycle'),
        Line2D([0], [0], color=styles['SOS'][0], linestyle=styles['SOS'][1], lw=styles['SOS'][2], label=fmt_legend('SOS', metrics)),
        Line2D([0], [0], color=styles['SOS+EVT'][0], linestyle=styles['SOS+EVT'][1], lw=styles['SOS+EVT'][2], label=fmt_legend('SOS+EVT', metrics)),
        Line2D([0], [0], color=styles['NLF+SMT'][0], linestyle=styles['NLF+SMT'][1], lw=styles['NLF+SMT'][2], label=fmt_legend('NLF+SMT', metrics)),
        Line2D([0], [0], color=styles['Zubov+SMT'][0], linestyle=styles['Zubov+SMT'][1], lw=styles['Zubov+SMT'][2], label=fmt_legend('Zubov+SMT', metrics)),
        Line2D([0], [0], color=styles['ICNN+SMT'][0], linestyle=styles['ICNN+SMT'][1], lw=styles['ICNN+SMT'][2], label=fmt_legend('ICNN+SMT', metrics)),
    ]

    if V_evt_dict_gram_grid is not None:
        legend_elements.append(
            Line2D([0], [0], color=styles['Dict-Gram+EVT'][0], linestyle=styles['Dict-Gram+EVT'][1], lw=styles['Dict-Gram+EVT'][2], label=fmt_legend('Dict-Gram+EVT', metrics))
        )

    ax.legend(handles=legend_elements, loc='upper right', 
              bbox_to_anchor=(1.0, 1.0),
              frameon=True, framealpha=0.95, 
              edgecolor='#cccccc', fancybox=True, fontsize=12, 
              handlelength=3.0) 
    
    # Save
    if not os.path.exists("plots"): os.makedirs("plots")
    save_path = "plots/paper_roa_comparison_final_rsgld.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")

if __name__ == "__main__":
    main()