import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme, kstest, probplot
import math
import random
import os

# ==========================================
# 0. Configuration & Helpers
# ==========================================
def create_dirs():
    os.makedirs('plots', exist_ok=True)
    print(" [INFO] Created 'plots/' and 'plot_traj/' directories.")

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f" [INFO] Seed set to {seed}")

# ==========================================
# 1. System Dynamics & Model 
# ==========================================
class DenseODE(nn.Module):
    def __init__(self, dim, seed, device='cpu'):
        super().__init__()
        self.dim = dim
        self.device = device
        
        A = np.zeros((dim, dim))
        for i in range(0, dim, 2):
            if i + 1 < dim:
                A[i, i+1] = 1.0; A[i+1, i] = -1.0; A[i+1, i+1] = -0.5
                
        np.random.seed(seed + dim)
        H = np.random.randn(dim, dim)
        Q, _ = np.linalg.qr(H)
        M = Q @ A @ Q.T
        self.matrix = torch.tensor(M, dtype=torch.float32, device=device)

    def vector_field(self, u):
        return u @ self.matrix.t()

def construct_monomials(u: torch.Tensor) -> torch.Tensor:
    return torch.cat([u, u**2], dim=1)

class GramMatrixLyapunov(nn.Module):
    def __init__(self, state_dim: int, device: str = 'cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.device = device
        self.feature_dim = 2 * state_dim 
        
        self.L_factor = nn.Parameter(torch.randn(self.feature_dim, self.feature_dim, device=device) * 0.01)
        with torch.no_grad():
            self.L_factor.add_(torch.eye(self.feature_dim, device=device) * 0.1)
        self.register_buffer('eye', torch.eye(self.feature_dim, device=device) * 1e-6)
    
    def get_Q(self) -> torch.Tensor:
        return self.L_factor @ self.L_factor.T + self.eye
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        z = construct_monomials(u)
        Q = self.get_Q()
        return torch.einsum('bf, fg, bg -> b', z, Q, z).view(-1, 1)

# ==========================================
# 2. Training the Lyapunov Function
# ==========================================
def train_lyapunov(physics, model, epochs=2000, batch_size=2048, lr=1e-3):
    """Trains the Gram matrix to ensure V_dot < 0 across the domain."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)
    
    print(f"\n==================================================")
    print(f" TRAINING LYAPUNOV FUNCTION ({epochs} Epochs)")
    print(f"==================================================")
    
    for epoch in range(epochs):
        u = torch.randn(batch_size, physics.dim, device=physics.device) * 1.5
        u.requires_grad_(True)
        
        V_val, V_dot = compute_V_Vdot(u, model, physics)
        
        stability_loss = torch.mean(torch.nn.functional.relu(V_dot + 0.05 * V_val))
        reg_loss = 1e-4 * torch.mean(V_val)
        
        loss = stability_loss + reg_loss
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)
        
        if epoch % 250 == 0 or epoch == epochs - 1:
            print(f" [Epoch {epoch:4d}] Loss: {loss.item():.6f} | Stability Violation: {stability_loss.item():.6f}")
            
    print(" [INFO] Training Complete.\n")

# ==========================================
# 3. PDE-Style SGLD Optimizer 
# ==========================================
def compute_V_Vdot(u, model, physics):
    if not u.requires_grad: u.requires_grad_(True)
    V_val = model(u)
    
    grads = torch.autograd.grad(outputs=V_val.sum(), inputs=u, create_graph=True)[0]
    f_u = physics.vector_field(u)
    
    V_dot = torch.sum(grads * f_u, dim=1)
    return V_val, V_dot

def boundary_attack_batch(physics, model, u_init_batch, rho, steps=1500, prune_freq=150, prune_ratio=0.2, T=1e-4, lr=0.01):
    u = u_init_batch.clone().detach()
    u.requires_grad = True
    
    for i in range(steps):
        if i > 0 and i % prune_freq == 0 and i < (steps - 100): 
            _, current_Vdots = compute_V_Vdot(u, model, physics)
            with torch.no_grad():
                n_prune = int(u.shape[0] * prune_ratio)
                if n_prune > 0:
                    vals, indices = torch.sort(current_Vdots)
                    worst_indices = indices[:n_prune]
                    
                    new_dirs = torch.randn(n_prune, physics.dim, device=u.device)
                    new_dirs = new_dirs / torch.norm(new_dirs, dim=1, keepdim=True)
                    scales = torch.rand(n_prune, 1, device=u.device) * (2.0 * np.sqrt(rho)) + (0.1 * np.sqrt(rho))
                    
                    u.data[worst_indices] = new_dirs * scales

        V_val, V_dot_batch = compute_V_Vdot(u, model, physics)
        
        V_c = V_val.squeeze()
        violation = (V_c - rho)**2
        
        loss = torch.sum(-V_dot_batch + (50.0 * violation))
        grads = torch.autograd.grad(loss, u)[0]
        
        with torch.no_grad():
            progress = i / steps
            current_lr = lr * (1.0 - progress) + 1e-4
            
            grads = torch.clamp(grads, -10.0, 10.0)
            
            sigma = math.sqrt(2.0 * current_lr * T)
            noise = torch.randn_like(u) * sigma
            u.data = u.data - (current_lr * grads) + noise
            
    _, final_Vdots = compute_V_Vdot(u, model, physics)
    return final_Vdots.detach()

# ==========================================
# 4. EVT Analysis 
# ==========================================
def analyze_evt_block_maxima(raw_samples, rho, block_size=50, confidence_level=0.95, filename_tag=""):
    data = np.array(raw_samples)
    data = data[data > -1e8] 
    
    jitter = np.random.normal(loc=0.0, scale=1e-6, size=data.shape)
    data = data + jitter
    
    n_samples = len(data)
    n_blocks = n_samples // block_size
    
    if n_blocks < 10:
        print(f" [EVT FAIL] Not enough blocks (n={n_blocks}). Need >= 10.")
        return 1.0, 0.0, False 

    reshaped_data = data[:n_blocks*block_size].reshape(n_blocks, block_size)
    block_maxima = np.max(reshaped_data, axis=1)

    sorted_bm = np.sort(block_maxima)
    gaps = np.diff(sorted_bm)
    median_gap = np.median(gaps) if np.median(gaps) > 1e-7 else 1e-7
    max_gap = np.max(gaps)
    gap_ratio = max_gap / median_gap
    is_fractured = gap_ratio > 20.0 and max_gap > 0.1 
    
    if is_fractured:
        print(f" >>> REJECTION REASON: Landscape Fracture Detected.")
        shape, loc, scale = 0.5, 0.0, 1.0 
        p_value = 0.0
        is_certified = False
        status_str = "REJECTED (Fracture)"
        status_color = 'purple'
        ep_lower_ci, ep_upper_ci = 0.0, 0.0
        mle_endpoint = np.inf 
    else:
        try:
            shape, loc, scale = genextreme.fit(block_maxima)
            mle_endpoint = loc + (scale / shape) if shape > 1e-4 else np.inf
        except Exception as e:
            print(f" >>> REJECTION REASON: MLE Optimization Failed ({e})")
            return 1.0, 0.0, False
            
        n_boots = 500
        boot_endpoints = []
        boot_shapes = [] 
        
        for _ in range(n_boots):
            # STATISTICALLY SOUND BOOTSTRAP: Resample the block maxima directly.
            # This avoids creating artificial point masses at the extreme right tail.
            resamp_bm = np.random.choice(block_maxima, size=len(block_maxima), replace=True)
            
            try:
                b_shape, b_loc, b_scale = genextreme.fit(resamp_bm)
                boot_shapes.append(b_shape)
                if b_shape > 1e-4: 
                    boot_endpoints.append(b_loc + (b_scale / b_shape))
                else:
                    boot_endpoints.append(10.0) 
            except: 
                continue
                
        boot_endpoints = np.array(boot_endpoints)
        boot_shapes = np.array(boot_shapes)
        
        alpha = 1.0 - confidence_level
        if len(boot_shapes) > 0:
            ep_upper_ci = np.percentile(boot_endpoints, (1 - alpha/2) * 100)
            ep_lower_ci = np.percentile(boot_endpoints, alpha/2 * 100) 
        else:
            ep_upper_ci = 100.0; ep_lower_ci = -100.0

        _, p_value = kstest(block_maxima, 'genextreme', args=(shape, loc, scale))
        
        fit_good = p_value > 0.05
        regime_safe = shape > 0 
        value_safe = ep_upper_ci < 0
        
        is_certified = fit_good and regime_safe and value_safe
        
        status_str = "CERTIFIED" if is_certified else "REJECTED"
        status_color = 'darkgreen' if is_certified else 'darkred'

        print(f"--------------------------------------------------")
        print(f" EVT Analysis for rho = {rho:.3f}")
        print(f" Shape (c): {shape:.4f} | KS p-val: {p_value:.4f}")
        print(f" Upper CI: {ep_upper_ci:.5f} | MLE End: {mle_endpoint:.5f}")
        print(f"--------------------------------------------------")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    ax_raw = axes[0, 0]; ax_block = axes[0, 1] 
    ax_qq = axes[1, 0]; ax_safe = axes[1, 1] 

    counts, bins, _ = ax_raw.hist(data, bins=100, density=True, alpha=0.6, color='purple')
    ax_raw.axvline(0, color='k', linestyle='--', linewidth=2)
    ax_raw.set_title(f"1. RAW Optimization Outcomes (N={n_samples})", fontweight='bold')
    if max(data) > 0: ax_raw.axvspan(0, max(data), color='red', alpha=0.1)

    if is_fractured:
        ax_block.hist(block_maxima, bins=30, density=True, alpha=0.4, color='gray')
        ax_block.text(0.5, 0.5, "LANDSCAPE FRACTURE", transform=ax_block.transAxes, ha='center', color='red', fontweight='bold')
    else:
        ax_block.hist(block_maxima, bins=20, density=True, alpha=0.4, color='steelblue')
        x_plot = np.linspace(min(block_maxima)-0.1, max(block_maxima)+0.1, 200)
        try: ax_block.plot(x_plot, genextreme.pdf(x_plot, shape, loc, scale), 'r-', linewidth=2.5)
        except: pass
    ax_block.set_title(f"2. Block Maxima\nStatus: {status_str}", color=status_color, fontweight='bold')

    if not is_fractured:
        probplot(block_maxima, dist=genextreme, sparams=(shape, loc, scale), plot=ax_qq)
        ax_qq.set_title(f"3. Q-Q Plot (KS p={p_value:.4f})")
    else:
        ax_qq.text(0.5, 0.5, "Q-Q Undefined", transform=ax_qq.transAxes, ha='center')

    def to_finite(val, fallback): return val if np.isfinite(val) else fallback
    vis_ep_upper = min(to_finite(ep_upper_ci, 10.0), 50.0)
    
    ax_safe.fill_betweenx([0, 10], -100, 0, color='green', alpha=0.15)
    ax_safe.fill_betweenx([0, 10], 0, 100, color='orange', alpha=0.15)
    ax_safe.fill_betweenx([-10, 0], -100, 100, color='red', alpha=0.15)
    ax_safe.axvline(0, color='r', linestyle='--', linewidth=2)
    ax_safe.axhline(0, color='k', linestyle='--', linewidth=2)
    
    if not is_fractured:
        mask = np.isfinite(boot_endpoints) & (boot_endpoints < 100)
        ax_safe.scatter(boot_endpoints[mask], boot_shapes[mask], c='purple', alpha=0.5, s=20)
        ax_safe.scatter([mle_endpoint], [shape], c='black', marker='x', s=100, zorder=10)
    
    ax_safe.set_ylim(-2, 2)
    ax_safe.set_xlim(min(block_maxima.min(), -1), max(vis_ep_upper, 1))
    ax_safe.set_title(f"4. Safety Map ({filename_tag})")
    
    plt.tight_layout()
    save_path = os.path.join('plots', f"evt_diagnostic_{filename_tag}_{rho:.4f}.png")
    # plt.savefig(save_path)
    # plt.close()
    
    prob_fail_mle = 1.0 if (mle_endpoint > 0 or is_fractured) else 0.0
    return prob_fail_mle, p_value, is_certified

# ==========================================
# 5. Certification & Search 
# ==========================================
def certify_with_evt(physics, model, rho, n_samples=5000, conf_level=0.99, tag="Search", force_plot=False):
    dirs = torch.randn(n_samples, physics.dim, device=physics.device)
    dirs = dirs / torch.norm(dirs, dim=1, keepdim=True)
    scales = torch.rand(n_samples, 1, device=physics.device) * (2.0 * np.sqrt(rho)) + (0.1 * np.sqrt(rho))
    u_batch = dirs * scales

    v_final_tensor = boundary_attack_batch(physics, model, u_batch, rho, steps=1500, prune_freq=150, T=1e-4)
    
    concrete_fail = torch.max(v_final_tensor).item() > 0
    if concrete_fail and not force_plot:
        print(f" [FAST REJECT] Concrete Counter-Example Found (Max={torch.max(v_final_tensor).item():.5f} > 0)")
        return False, 1.0
        
    maxima_list = v_final_tensor.cpu().numpy().tolist()
    
    prob_fail, ks_p, is_cert = analyze_evt_block_maxima(
        maxima_list, rho, block_size=100, 
        confidence_level=conf_level, filename_tag=tag
    )
    
    if concrete_fail: 
        return False, 1.0
        
    return is_cert, prob_fail

def find_robust_roa(physics, model, start_rho=0.01, max_rho=5.0, confidence=0.99):
    low, high = start_rho, max_rho
    best_rho = start_rho
    print(f"\n==================================================")
    print(f" MAIN QUEST: Searching for {physics.dim}D ROA")
    print(f"==================================================")
    
    for i in range(5): 
        rho = (low + high) / 2.0
        print(f"\n>>> Checking rho = {rho:.3f} ...")
        
        is_safe, _ = certify_with_evt(physics, model, rho, n_samples=5000, conf_level=confidence,
                                      tag=f"Search_{i}", force_plot=False)
        
        if is_safe:
            best_rho = rho; low = rho
            print(f" [RESULT] rho={rho:.3f} ACCEPTED")
        else:
            high = rho
            print(f" [RESULT] rho={rho:.3f} REJECTED")
            
    return best_rho

# ==========================================
# 6. Execution 
# ==========================================
if __name__ == "__main__":
    # create_dirs() 
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dim = 500
    physics = DenseODE(dim=dim, seed=42, device=device)
    model = GramMatrixLyapunov(state_dim=dim, device=device).to(device)
    
    # Warm start LQR initialization
    try:
        import scipy.linalg
        Q_cost = np.eye(dim)
        A_np = physics.matrix.cpu().numpy()
        P = scipy.linalg.solve_continuous_lyapunov(A_np.T, -Q_cost)
        if np.all(np.linalg.eigvals(P) > 0):
            L_lqr = np.linalg.cholesky(P)
            with torch.no_grad():
                model.L_factor[:dim, :dim] = torch.tensor(L_lqr, dtype=torch.float32, device=device)
            print(" [INFO] Model initialized with LQR solution.")
    except Exception as e:
        print(f" [WARNING] LQR Init failed, using random: {e}")
        
    train_lyapunov(physics, model, epochs=2000, batch_size=2048, lr=1e-3)

    confidence = 0.9999 # 99.99% confidence for EVT certification

    c_final = find_robust_roa(physics, model, start_rho=0.01, max_rho=3.0, confidence=confidence)
    print(f"\n>>> FINAL CERTIFIED ROA ({dim}D): rho = {c_final:.4f}")