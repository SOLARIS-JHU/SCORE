import torch
import numpy as np
import math
import os
import sys

from commons import VanDerPol, ODEGramMatrixLyapunov
from verify_evt_new import certify_with_evt, compute_V_Vdot

def get_ground_truth_gamma_volume(physics, model, rho, rho_core=0.05, grid_size=4000):
    """
    Computes the exact ground truth maximum of V_dot strictly inside the donut volume 
    (rho_core <= V(x) <= rho) using an ultra-dense chunked 2D grid search.
    """
    model.eval()
    
    # 1. Create a massive dense grid over the expected state space
    x = torch.linspace(-3.5, 3.5, grid_size, device=physics.device)
    y = torch.linspace(-3.5, 3.5, grid_size, device=physics.device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    u_grid = torch.stack([X.flatten(), Y.flatten()], dim=1)
    total_points = u_grid.shape[0]
    
    print(f" [GROUND TRUTH] Scanning {total_points:,} grid points inside the donut volume...")
    
    # Process in chunks to prevent GPU Out-of-Memory errors
    chunk_size = 2_000_000 
    max_vdot = -float('inf')
    points_in_donut = 0
    
    for i in range(0, total_points, chunk_size):
        u_chunk = u_grid[i:i+chunk_size].clone()
        u_chunk.requires_grad_(True)
        
        V_val, V_dot, _ = compute_V_Vdot(u_chunk, model, physics)
        V_val = V_val.squeeze()
        
        # 2. Filter points to only those exactly inside the donut
        valid_mask = (V_val <= rho) & (V_val >= rho_core)
        points_in_donut += valid_mask.sum().item()
        
        if valid_mask.any():
            vdot_valid = V_dot[valid_mask]
            chunk_max = torch.max(vdot_valid).item()
            if chunk_max > max_vdot:
                max_vdot = chunk_max
                
    print(f" [GROUND TRUTH] Found {points_in_donut:,} points valid within the donut bounds.")
    return max_vdot

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
    
    test_rho = 1.6277

    rho_core = 0.05
    print(f"\n--- Running Ground Truth Dense Grid Search for Donut: {rho_core} <= rho <= {test_rho} ---")
    
    # Compute ground truth
    gt_gamma = get_ground_truth_gamma_volume(physics, model, test_rho, rho_core=rho_core, grid_size=4000)
    print(f" [GROUND TRUTH] Exact Max V_dot in volume (gamma*): {gt_gamma:.6f}")
    
    # 2. Run EVT Certification over multiple random seeds to check the Confidence Interval
    print("\n--- Validating EVT Confidence Interval (99%) ---")
    print(f"{'Seed':<10} | {'EVT Upper Bound (CI)':<20} | {'Condition (CI > GT?)':<20} | {'Failure Reason':<20}")
    print("-" * 75)
    
    # Run 500 certifications
    test_seeds = list(np.arange(0, 500))
    
    # Trackers for the final statistics
    valid_count = 0
    invalid_count = 0
    
    for seed in test_seeds:
        # We must seed torch and numpy for the SGLD sampling to vary
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Suppress EVT prints for a clean table
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        # Capture the 4 return values from the updated certify_with_evt function
        _, ci_upper, is_safe, reason = certify_with_evt(
            physics, 
            model, 
            test_rho, 
            n_samples=10000, 
            block_size=100,
            steps=1000,
            tag="val"
        )
        
        # Restore prints
        sys.stdout = old_stdout
        
        # Check if the bound is conservative compared to our new volume ground-truth
        is_valid = ci_upper >= (gt_gamma - 1e-5)
        
        if is_valid:
            valid_str = "VALID (Conservative)"
            valid_count += 1
        else:
            valid_str = "INVALID (Underestimated)"
            invalid_count += 1
            
        print(f"{seed:<10} | {ci_upper:<20.6f} | {valid_str:<20} | {reason:<20}")
        
    # --- Final Aggregation and Output ---
    total_seeds = len(test_seeds)
    success_rate = (valid_count / total_seeds) * 100
    
    print("\n" + "="*75)
    print("FINAL EVT BENCHMARK RESULTS")
    print("="*75)
    print(f"Total Seeds Tested: {total_seeds}")
    print(f"Valid Bounds (Conservative): {valid_count}")
    print(f"Invalid Bounds (Underestimated): {invalid_count}")
    print(f"Empirical Success Rate: {success_rate:.2f}%")
    print("="*75)