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
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)
    
    print(f"\n==================================================")
    print(f" TRAINING LYAPUNOV FUNCTION ({epochs} Epochs)")
    print(f"==================================================")
    
    for epoch in range(epochs):
        u = torch.randn(batch_size, physics.dim, device=physics.device) * 1.5
        u.requires_grad_(True)
        
        V_val = model(u)
        grads = torch.autograd.grad(outputs=V_val.sum(), inputs=u, create_graph=True)[0]
        f_u = physics.vector_field(u)
        V_dot = torch.sum(grads * f_u, dim=1).unsqueeze(1)
        
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
# 3. Softplus-Squared RSGLD (Smooth Volume)
# ==========================================
def compute_V_Vdot(u, model, physics):
    if not u.requires_grad: u.requires_grad_(True)
    V_val = model(u)
    grads = torch.autograd.grad(outputs=V_val.sum(), inputs=u, create_graph=True)[0]
    f_u = physics.vector_field(u)
    V_dot = torch.sum(grads * f_u, dim=1)
    return V_val, V_dot

def volume_attack_batch(physics, model, u_init_batch, rho, rho_core=0.01, steps=1500, prune_freq=150, prune_ratio=0.2, T=1e-4, lr=0.01):
    u = u_init_batch.clone().detach()
    u.requires_grad = True
    
    for i in range(steps):
        # A. Pruning (Matches the original script's aggressive outside-in teleportation)
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

        # B. SGLD Gradient Step (Maximizing V_dot + Smooth Volume Penalties)
        u.requires_grad_(True)
        V_val, V_dot_batch = compute_V_Vdot(u, model, physics)
        V_c = V_val.squeeze()
        
        # THE ENGINEERING TRICK: Softplus Squared.
        # Perfectly replicates (V_c - rho)**2 outside the boundary, but melts to 0 inside.
        # Beta=10.0 ensures C-infinity smoothness, preserving the EVT Weibull Hessian.
        out_violation = torch.nn.functional.softplus(V_c - rho, beta=10.0)**2
        in_violation = torch.nn.functional.softplus(rho_core - V_c, beta=10.0)**2
        
        loss = torch.sum(-V_dot_batch + (50.0 * out_violation) + (50.0 * in_violation))
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
def analyze_evt_block_maxima(raw_samples, rho, block_size=50, confidence_level=0.99, filename_tag=""):
    data = np.array(raw_samples)
    data = data[data > -1e8] 
    
    jitter = np.random.normal(loc=0.0, scale=1e-7, size=data.shape)
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
        mle_endpoint = np.inf 
        ep_upper_ci = np.inf
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
        else:
            ep_upper_ci = 100.0

        _, p_value = kstest(block_maxima, 'genextreme', args=(shape, loc, scale))
        
        fit_good = p_value > 0.05
        regime_safe = shape > 0 
        value_safe = ep_upper_ci < -1e-6
        
        is_certified = fit_good and regime_safe and value_safe
        
        print(f"--------------------------------------------------")
        print(f" EVT Analysis for rho = {rho:.3f}")
        print(f" Shape (c): {shape:.4f} | KS p-val: {p_value:.4f}")
        print(f" Upper CI: {ep_upper_ci:.2e} | MLE End: {mle_endpoint:.2e}")
        print(f"--------------------------------------------------")

    prob_fail_mle = 1.0 if (mle_endpoint > 0 or is_fractured) else 0.0
    return prob_fail_mle, p_value, is_certified

# ==========================================
# 5. Certification & Adaptive Search 
# ==========================================
def certify_with_evt(physics, model, rho, n_samples=5000, block_size=100, steps=1500, conf_level=0.99, tag="Search"):
    dirs = torch.randn(n_samples, physics.dim, device=physics.device)
    dirs = dirs / torch.norm(dirs, dim=1, keepdim=True)
    # Matches original script initialization (starting outside and being pulled in)
    scales = torch.rand(n_samples, 1, device=physics.device) * (2.0 * np.sqrt(rho)) + (0.1 * np.sqrt(rho))
    u_init = dirs * scales

    # Single-phase run, exactly like lyapunov_core
    v_final_tensor = volume_attack_batch(physics, model, u_init, rho, rho_core=0.01, steps=steps, prune_freq=150, prune_ratio=0.2, T=1e-4)
    
    concrete_fail = torch.max(v_final_tensor).item() > 0
    if concrete_fail:
        print(f" [FAST REJECT] Concrete Counter-Example Found (Max={torch.max(v_final_tensor).item():.5e} > 0)")
        return False, "CONCRETE_VIOLATION"
        
    maxima_list = v_final_tensor.cpu().numpy().tolist()
    
    prob_fail, ks_p, is_cert = analyze_evt_block_maxima(
        maxima_list, rho, block_size=block_size, 
        confidence_level=conf_level, filename_tag=tag
    )
    
    reason = "NONE" if is_cert else "STATISTICAL_FAILURE"
    return is_cert, reason

def find_robust_roa(physics, model, start_rho=0.01, max_rho=5.0, confidence=0.99):
    low, high = start_rho, max_rho
    best_rho = start_rho
    print(f"\n==================================================")
    print(f" MAIN QUEST: Searching for {physics.dim}D ROA (Smooth Volume)")
    print(f"==================================================")
    
    for i in range(10): 
        rho = (low + high) / 2.0
        
        current_samples = 5000
        current_block = 100
        current_steps = 1500
        max_retries = 2
        is_safe_overall = False
        
        for attempt in range(max_retries):
            print(f"\n>>> Checking rho={rho:.3f} (Attempt {attempt+1}/{max_retries} | N={current_samples}, m={current_block}, steps={current_steps})")
            
            is_safe, reason = certify_with_evt(
                physics, model, rho, 
                n_samples=current_samples, block_size=current_block, steps=current_steps,
                conf_level=confidence, tag=f"Search_rho{rho:.3f}_att{attempt}"
            )
            
            if is_safe:
                is_safe_overall = True
                break
            elif reason == "CONCRETE_VIOLATION":
                break
            elif reason == "STATISTICAL_FAILURE":
                print("   [TUNE] Statistical failure. Automatically increasing parameters...")
                current_samples = int(current_samples * 2.0)  
                current_block = int(current_block * 1.5)      
                current_steps = int(current_steps * 1.2)      
                
                if current_samples > 25000:
                    print("   [TUNE] Reached memory limits. Aborting retries.")
                    break
                    
        if is_safe_overall:
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

    c_final = find_robust_roa(physics, model, start_rho=0.01, max_rho=3.5, confidence=confidence)
    print(f"\n>>> FINAL CERTIFIED ROA ({dim}D): rho = {c_final:.4f}")