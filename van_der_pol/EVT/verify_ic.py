import torch
import numpy as np
import math
import os

from commons import VanDerPol, ODEGramMatrixLyapunov
from verify_evt import certify_with_evt, compute_V_Vdot

def get_ground_truth_gamma(physics, model, rho, num_rays=10_000_000):
    """
    Computes the exact ground truth maximum of V_dot on the level set V(x) = rho
    using ultra-dense polar ray-casting and Newton projection.
    """
    model.eval()
    
    # 1. Shoot `num_rays` uniformly across [0, 2pi]
    theta = torch.linspace(0, 2 * math.pi, num_rays, device=physics.device).view(-1, 1)
    dir_vecs = torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)
    
    # 2. Guess the radius (since V is roughly quadratic, V(r*dir) ~= r^2 * V(dir))
    with torch.no_grad():
        V_dir = model(dir_vecs)
        r_init = torch.sqrt(rho / V_dir)
    
    # 3. Initialize points and prepare for gradient tracking
    u = (dir_vecs * r_init).clone().detach()
    u.requires_grad_(True)
    
    # 4. Exact Newton-Raphson Projection to snap perfectly onto V(x) = rho
    for _ in range(20): # 20 steps should guarantee machine-precision convergence
        V_curr = model(u)
        err = V_curr - rho
        
        grads = torch.autograd.grad(V_curr.sum(), u)[0]
        grad_sq_norm = torch.sum(grads**2, dim=1, keepdim=True) + 1e-8
        
        step = (err / grad_sq_norm) * grads
        u.data = u.data - step
        
    # 5. Filter out any points that somehow didn't converge (safety check)
    with torch.no_grad():
        V_final = model(u)
        valid_mask = (torch.abs(V_final - rho) < 1e-3).flatten()
        u_valid = u[valid_mask]
        
    if len(u_valid) < num_rays * 0.9:
        print(f" [WARNING] Only {len(u_valid)}/{num_rays} points converged to the boundary.")

    # 6. Compute V_dot on these highly dense, exact boundary points
    u_valid.requires_grad_(True)
    _, V_dot, _ = compute_V_Vdot(u_valid, model, physics)
    
    # The absolute maximum is our Ground Truth gamma*
    gamma_star = torch.max(V_dot).item()
    
    return gamma_star


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load the trained model
    physics = VanDerPol(mu=1.0, device=device)
    model = ODEGramMatrixLyapunov(state_dim=2, feature_dim=10, device=device).to(device)
    
    model_path = "models/lyapunov_model.pth"
    if not os.path.exists(model_path):
        print(" [ERROR] Model not found. Run train_model.py first.")
        exit()
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    test_rho = 1.6023
    print(f"\n--- Running Ground Truth Dense Grid Search for rho = {test_rho} ---")
    
    # Compute ground truth
    gt_gamma = get_ground_truth_gamma(physics, model, test_rho, num_rays=10_000_000)
    print(f" [GROUND TRUTH] Exact Max V_dot (gamma*): {gt_gamma:.6f}")
    
    # 2. Run EVT Certification over multiple random seeds to check the Confidence Interval
    print("\n--- Validating EVT Confidence Interval (99%) ---")
    print(f"{'Seed':<10} | {'EVT Upper Bound (CI)':<20} | {'Condition (CI > GT?)':<20}")
    print("-" * 55)
    
    test_seeds = [1, 2, 3, 4, 5]
    
    for seed in test_seeds:
        # We must seed torch and numpy for the SGLD sampling to vary
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Suppress EVT prints for a clean table
        import sys, os
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        # Run standard EVT certification
        _, ci_upper, is_safe = certify_with_evt(physics, model, test_rho, n_samples=2000, tag="val")
        
        # Restore prints
        sys.stdout = old_stdout
        
        is_valid = ci_upper >= (gt_gamma - 1e-5)
        valid_str = "VALID (Conservative)" if is_valid else "INVALID (Underestimated)"
        
        print(f"{seed:<10} | {ci_upper:<20.6f} | {valid_str:<20}")
        