import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import genextreme, probplot, kstest
import math
import random
import os
import warnings

from commons import VanDerPol, ODEGramMatrixLyapunov

# ==========================================
# 0. Configuration & Helpers
# ==========================================
TIME_REVERSE = False  

def create_dirs():
    os.makedirs('plots', exist_ok=True)
    os.makedirs('plot_traj', exist_ok=True)
    print(" [INFO] Created 'plots/' and 'plot_traj/' directories.")

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f" [INFO] Seed set to {seed}")

# ==========================================
# 1. Derivatives & Sampling
# ==========================================
def compute_V_Vdot(u, model, physics, time_reverse=TIME_REVERSE):
    if not u.requires_grad: u.requires_grad_(True)
    
    # 1. Compute V(u)
    V_val = model(u)
    
    # 2. Compute gradients dV/dx
    grads = torch.autograd.grad(outputs=V_val.sum(), inputs=u, create_graph=True)[0]
    
    # 3. Compute Vector Field f(u)
    f_u = physics.vector_field(u)
    
    if time_reverse:
        f_u = -f_u
        
    # 4. Lie Derivative: V_dot = grad * f
    V_dot = torch.sum(grads * f_u, dim=1)
    
    return V_val, V_dot, f_u

def reflect_on_boundaries(u, model, rho, rho_core=0.05, max_iter=5):
    """
    Reflect particles back into the 'donut' volume: rho_core <= V(x) <= rho.
    Prevents SGLD from collapsing into the trivial origin where V_dot = 0.
    """
    with torch.no_grad():
        for _ in range(max_iter):
            with torch.enable_grad():
                u.requires_grad_(True)
                V_curr = model(u).squeeze()
                
                # Check both outer bound (too big) and inner bound (too close to origin)
                out_mask = V_curr > rho
                in_mask = V_curr < rho_core
                
                violation_mask = out_mask | in_mask
                
                if not violation_mask.any():
                    break 
                
                # Define the target V based on which boundary was crossed
                v_target = V_curr.clone()
                v_target[out_mask] = rho
                v_target[in_mask] = rho_core
                
                v_err = V_curr - v_target
                
                # Compute gradient of V w.r.t u
                grad_V = torch.autograd.grad(V_curr.sum(), u)[0]
            
            # Newton Update
            grad_sq_norm = torch.sum(grad_V**2, dim=1, keepdim=True)
            grad_sq_norm = torch.clamp(grad_sq_norm, min=1e-6) 
            
            step = (v_err.view(-1, 1) / grad_sq_norm) * grad_V
            step = torch.clamp(step, -0.5, 0.5) 
            
            # Apply reflection ONLY to violating particles
            u.data[violation_mask] = u.data[violation_mask] - step[violation_mask]
            
    return u

def volume_attack_batch(physics, model, u_init_batch, rho, rho_core=0.05, steps=1000, 
                          prune_freq=100, T=1e-2, lr=0.01, 
                          sampling_mode=False):
    """
    Performs Reflected SGLD strictly inside the donut volume.
    """
    u = u_init_batch.clone().detach()
    u.requires_grad = True
    
    # --- PHASE 0: Initial Reflection ---
    u = reflect_on_boundaries(u, model, rho, rho_core, max_iter=10)
        
    # --- MAIN LOOP ---
    for i in range(steps):
        
        # A. Pruning
        if not sampling_mode and i > 0 and i % prune_freq == 0 and i < (steps - 50):
            _, current_Vdots, _ = compute_V_Vdot(u, model, physics)
            with torch.no_grad():
                n_prune = int(u.shape[0] * 0.5) 
                vals, indices = torch.sort(current_Vdots)
                safe_indices = indices[:n_prune]       
                dangerous_indices = indices[-n_prune:]   
                
                noise = torch.randn_like(u.data[safe_indices]) * 0.2
                u.data[safe_indices] = u.data[dangerous_indices] + noise

        # B. SGLD Gradient Step
        u.requires_grad_(True)
        _, V_dot_batch, _ = compute_V_Vdot(u, model, physics)
        
        loss = torch.sum(-V_dot_batch) 
        grads = torch.autograd.grad(loss, u)[0]
        
        with torch.no_grad():
            current_lr = lr * (1.0 - i/steps)
            sigma = math.sqrt(2.0 * current_lr * T)
            noise = torch.randn_like(u) * sigma
            
            # FIX: Hard clamp on gradients to prevent the "0/5000 remaining" NaN explosions
            grads = torch.clamp(grads, -10.0, 10.0)
            
            u.data = u.data - (current_lr * grads) + noise
            u.data.clamp_(-4.0, 4.0)

        # C. Reflected Boundary (Outer + Inner)
        u = reflect_on_boundaries(u, model, rho, rho_core, max_iter=2)

    # --- PHASE 3: Filtering & Return ---
    u = reflect_on_boundaries(u, model, rho, rho_core, max_iter=5) 
    
    V_final = model(u).squeeze()
    _, final_Vdots, _ = compute_V_Vdot(u, model, physics)
    
    # Validation Masks (Updated for Donut)
    mask_finite = torch.isfinite(final_Vdots).flatten()
    mask_inside_donut = (V_final <= rho + 1e-4) & (V_final >= rho_core - 1e-4)
    valid_mask = mask_finite & mask_inside_donut
    
    if valid_mask.sum() < (u.shape[0] * 0.5):
        print(f"   [WARNING] High sample loss: {valid_mask.sum()}/{u.shape[0]} remaining.")
    
    return final_Vdots[valid_mask].detach(), u[valid_mask].detach()

# ==========================================
# 2. EVT Analysis & Bootstrapping
# ==========================================
def analyze_evt_block_maxima(raw_samples, rho, block_size=50, filename_tag=""):
    data = np.array(raw_samples)
    data = data[np.isfinite(data)]
    
    # Add jitter
    data = data + np.random.normal(0, 1e-7, size=data.shape)
    
    n_samples = len(data)
    n_blocks = n_samples // block_size
    if n_blocks < 10: 
        print(f"   [REJECT] Insufficient blocks (n={n_blocks} < 10).")
        return 1.0, 0.0, False 

    reshaped = data[:n_blocks*block_size].reshape(n_blocks, block_size)
    block_maxima = np.max(reshaped, axis=1)

    # --- TEST 1: Landscape Fracture Test ---
    sorted_bm = np.sort(block_maxima)
    gaps = np.diff(sorted_bm)
    if len(gaps) > 0:
        median_gap = np.median(gaps) if np.median(gaps) > 1e-7 else 1e-7
        max_gap = np.max(gaps)
        gap_ratio = max_gap / median_gap
        is_fractured = (gap_ratio > 20.0) and (max_gap > 0.1)
    else:
        is_fractured = False

    if is_fractured:
        print(f"   [REJECT] Fracture detected (Gap Ratio: {gap_ratio:.1f}).")
        shape, loc, scale = 0.5, 0.0, 1.0 
        upper_bound = np.inf
        ci_upper = np.inf
        p_value = 0.0
    else:
        try:
            # Fit Original Data
            shape, loc, scale = genextreme.fit(block_maxima)
            if shape > 0: 
                upper_bound = loc + (scale / shape)
            else:
                upper_bound = np.inf 
        except Exception as e:
            print(f"   [REJECT] MLE Optimization Failed: {e}")
            return 1.0, 0.0, False

        # --- TEST 2: Kolmogorov-Smirnov (KS) Test ---
        _, p_value = kstest(block_maxima, 'genextreme', args=(shape, loc, scale))

        # --- TEST 3: Bootstrap Confidence Interval ---
        n_boot = 200     # Number of bootstrap resamples 
        alpha = 0.01     # 99% Confidence Interval
        boot_bounds = []

        # Suppress warnings to prevent terminal spam from failing MLE fits
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(n_boot):
                boot_sample = np.random.choice(block_maxima, size=n_blocks, replace=True)
                try:
                    # Provide original MLE params as starting point for faster convergence
                    sh_b, loc_b, sc_b = genextreme.fit(boot_sample, shape, loc=loc, scale=scale)
                    if sh_b > 0:
                        boot_bounds.append(loc_b + (sc_b / sh_b))
                    else:
                        boot_bounds.append(np.inf)
                except Exception:
                    continue # Skip if MLE fails for this specific resample

        if len(boot_bounds) > 0:
            # Get the upper (1-alpha) percentile of the computed upper bounds
            ci_upper = np.percentile(boot_bounds, (1 - alpha) * 100)
        else:
            ci_upper = np.inf

    # --- Determine Certification Status & Print Reason ---
    reasons = []
    
    # 1. Check Tail
    if shape <= 0:
        reasons.append(f"Heavy Tail (shape={shape:.2f} <= 0)")
        
    # 2. Check Upper Bound (Strict safety via Bootstrap CI)
    if ci_upper >= -1e-4:
        reasons.append(f"Unsafe CI Bound (CI_max={ci_upper:.4f} >= 0)")
        
    # 3. Check Goodness of Fit (KS)
    if p_value <= 0.05:
        reasons.append(f"Bad Fit (p={p_value:.4f} < 0.05)")
        
    # 4. Check Fracture
    if is_fractured:
        reasons.append("Fractured Landscape")

    is_certified = (len(reasons) == 0)

    if is_certified:
        print(f"   [PASS] Certified! (Est. Max: {upper_bound:.4f}, 95% CI Max: {ci_upper:.4f}, p: {p_value:.3f})")
    else:
        # Give some helpful info even on failure
        bound_str = f"CI: {ci_upper:.4f}" if not np.isinf(ci_upper) else "CI: inf"
        print(f"   [REJECT] {', '.join(reasons)} | Point Est: {upper_bound:.4f}, {bound_str}")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(block_maxima, bins=15, density=True, alpha=0.6, color='b')
    axes[0].axvline(x=0, color='r', linestyle='--', label='Stability Boundary')
    if is_fractured:
        axes[0].text(0.5, 0.5, "FRACTURED", transform=axes[0].transAxes, 
                     color='red', fontweight='bold', ha='center')
    elif shape > 0:
        axes[0].axvline(x=upper_bound, color='g', linestyle='-', label=f'Est Max: {upper_bound:.4f}')
        axes[0].axvline(x=ci_upper, color='orange', linestyle=':', label=f'95% CI: {ci_upper:.4f}')
        
    axes[0].legend()
    status_str = "SAFE" if is_certified else "UNSAFE"
    if is_fractured: status_str = "FRACTURED"
    elif p_value <= 0.05: status_str = "BAD FIT"
    
    axes[0].set_title(f"Block Maxima (rho={rho:.3f})\nStatus: {status_str}")

    # Q-Q Plot
    if not is_fractured:
        probplot(block_maxima, dist=genextreme, sparams=(shape, loc, scale), plot=axes[1])
        axes[1].set_title(f"Q-Q Plot (KS p-val: {p_value:.4f})")
    else:
        axes[1].text(0.5, 0.5, "Skipped (Fracture)", ha='center')

    # plt.savefig(os.path.join('plots', f"evt_{filename_tag}_{rho:.3f}.png"))
    plt.close()
    
    return (0.0 if is_certified else 1.0), ci_upper, is_certified

# ==========================================
# 3. Search Logic
# ==========================================
def certify_with_evt(physics, model, rho, n_samples=2000, block_size=100, steps=1000, tag="Search"):
    """
    Runs the RSGLD EVT pipeline with configurable parameters.
    Returns: (dummy_val, ci_upper, is_certified, failure_reason)
    """
    u_init = (torch.rand(n_samples, 2, device=physics.device) * 5.0) - 2.5
    
    # Burn-in phase (scale steps proportionally if requested)
    burn_in_steps = max(200, int(steps * 0.2))
    _, u_burned_in = volume_attack_batch(
        physics, model, u_init, rho, 
        steps=burn_in_steps, prune_freq=50,  
        sampling_mode=False 
    )
    
    # Main sampling phase
    v_dot_final, _ = volume_attack_batch(
        physics, model, u_burned_in, rho, 
        steps=steps, 
        sampling_mode=True
    )
    
    if len(v_dot_final) == 0: 
        return 1.0, 1.0, False, "NO_SAMPLES"
    
    # 1. Concrete Violation Check
    if torch.max(v_dot_final).item() > 0:
        print(f"   [REJECT] Concrete Counter-example found (V_dot > 0)")
        return 1.0, 1.0, False, "CONCRETE_VIOLATION"
        
    maxima_list = v_dot_final.cpu().numpy().tolist()
    
    # 2. Statistical EVT Check
    val, ci, is_safe = analyze_evt_block_maxima(maxima_list, rho, block_size=block_size, filename_tag=tag)
    
    failure_reason = "NONE" if is_safe else "STATISTICAL_FAILURE"
    return val, ci, is_safe, failure_reason

def find_robust_roa(physics, model):
    low, high = 0.01, 3.0
    best_rho = 0.01
    
    print(f"\n[INFO] Starting Binary Search for ROA with Adaptive Tuning...")
    for i in range(20):
        rho = (low + high) / 2.0
        
        # Base parameters for the first attempt
        current_samples = 2000
        current_block = 100
        current_steps = 1000
        max_retries = 3
        
        is_safe_overall = False
        
        for attempt in range(max_retries):
            print(f"\nChecking rho={rho:.3f} (Attempt {attempt+1}/{max_retries} | N={current_samples}, m={current_block}, steps={current_steps})")
            
            _, est_max, is_safe, reason = certify_with_evt(
                physics, model, rho, 
                n_samples=current_samples, 
                block_size=current_block, 
                steps=current_steps,
                tag=f"iter_{i}_attempt_{attempt}"
            )
            
            if is_safe:
                is_safe_overall = True
                break # Success! Move to a larger volume.
                
            elif reason == "CONCRETE_VIOLATION":
                # Math strictly forbids this volume. Do not retry.
                break 
                
            elif reason == "STATISTICAL_FAILURE":
                # The math might be safe, but the statistics were too blurry.
                # Automatically tune up parameters for the next attempt.
                print("   [TUNE] Statistical failure. Automatically increasing parameters...")
                current_samples = int(current_samples * 2.5)  # Boost sample pool
                current_block = int(current_block * 1.5)      # Boost block size
                current_steps = int(current_steps * 1.5)      # Boost mixing time
                
                # Failsafe: Prevent GPU Out-of-Memory (adjust based on your VRAM)
                if current_samples > 15000:
                    print("   [TUNE] Reached memory limits. Aborting retries for this rho.")
                    break
        
        # Standard Binary Search Update
        if is_safe_overall:
            best_rho = rho
            low = rho
        else:
            high = rho
            
    return best_rho

# ==========================================
# 4. Visualization
# ==========================================
def plot_representative_certification(physics, model, rho_certified):
    model.eval()
    
    x = np.linspace(-2.0, 2.0, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    pts_tensor = torch.tensor(pts, dtype=torch.float32, device=physics.device)
    pts_tensor.requires_grad_(True)
    
    V_val, V_dot_val, f_val = compute_V_Vdot(pts_tensor, model, physics, time_reverse=TIME_REVERSE)
    
    V_np = V_val.detach().cpu().numpy().reshape(X.shape)
    V_dot_np = V_dot_val.detach().cpu().numpy().reshape(X.shape)
    f_np = f_val.detach().cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    ax[0].contourf(X, Y, V_np, levels=50, cmap='Blues', alpha=0.3)
    ax[0].streamplot(X, Y, 
                     f_np[:,0].reshape(X.shape), 
                     f_np[:,1].reshape(X.shape), 
                     density=1.2, 
                     color=(0.5, 0.5, 0.5, 0.6), 
                     linewidth=0.5, 
                     arrowsize=0.8)

    u0 = torch.tensor([[0.01, 0.01]], device=physics.device)
    times, traj = physics.solve(u0, T=30.0, dt=0.01)
    ax[0].plot(traj[1500:,0], traj[1500:,1], 'k--', linewidth=2.0, alpha=0.8, label='Physical Limit Cycle')

    ax[0].contour(X, Y, V_np, levels=[rho_certified], colors='green', linewidths=3.0)
    
    ax[0].set_title(f"Certified ROA (Green) vs Trajectories\n(rho={rho_certified:.3f})")
    ax[0].legend(loc='upper right')
    ax[0].set_xlim([-3, 3])
    ax[0].set_ylim([-3, 3])

    vmin, vmax = V_dot_np.min(), V_dot_np.max()
    norm = mcolors.TwoSlopeNorm(vmin=max(vmin, -1.0), vcenter=0., vmax=min(vmax, 1.0))
    c2 = ax[1].contourf(X, Y, V_dot_np, levels=50, cmap='RdYlGn_r', norm=norm)
    plt.colorbar(c2, ax=ax[1], label='V_dot (Green=Stable)')
    
    ax[1].contour(X, Y, V_dot_np, levels=[0], colors='black', linewidths=2.5, linestyles='--')
    ax[1].plot(traj[1500:,0], traj[1500:,1], 'w:', linewidth=2.0, alpha=0.7, label='Limit Cycle')
    ax[1].contour(X, Y, V_np, levels=[rho_certified], colors='cyan', linewidths=3.0)

    ax[1].set_title("Verification: ROA (Cyan) inside Stability Region")
    ax[1].legend(loc='upper right')
    ax[1].set_xlim([-3, 3])
    ax[1].set_ylim([-3, 3])

    plt.tight_layout()
    save_path = os.path.join('plot_traj', 'certified_landscape.png')
    # plt.savefig(save_path)
    # print(f" [INFO] Saved representative plot to {save_path}")

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    # create_dirs()
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    physics = VanDerPol(mu=1.0, device=device)
    model = ODEGramMatrixLyapunov(state_dim=2, feature_dim=10, device=device).to(device)
    
    model_path = "models/lyapunov_model.pth"
    if os.path.exists(model_path):
        print(f" [INFO] Loading model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(" [ERROR] Please run train_model.py first to generate lyapunov_model.pth")
        exit()
            
    final_rho = find_robust_roa(physics, model)
    print(f"\n >>> FINAL CERTIFIED ROA: rho = {final_rho}")
    
    plot_representative_certification(physics, model, final_rho)