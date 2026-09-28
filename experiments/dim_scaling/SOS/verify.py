import numpy as np
import time
from pydrake.all import (
    Variable,
    Polynomial,
    MathematicalProgram,
    Solve,
    Variables,
)
from scipy.linalg import solve_continuous_lyapunov

# --------------------------------------------------------------------------
# 1. Physics: The Rotated Damped Oscillator
# --------------------------------------------------------------------------

def get_dense_system_matrix(dim):
    """Dense stable system: Block-diagonal damped oscillator rotated by Q."""
    A = np.zeros((dim, dim))
    for i in range(0, dim, 2):
        A[i, i+1] = 1.0
        A[i+1, i] = -1.0
        A[i+1, i+1] = -0.5
        
    rng = np.random.default_rng(seed=dim) 
    H = rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(H)
    
    M = Q @ A @ Q.T
    return M

def get_lyapunov_matrix(dim):
    """Solve Lyapunov equation for domain constraint."""
    A = get_dense_system_matrix(dim)
    Q = np.eye(dim)
    
    try:
        P = solve_continuous_lyapunov(A.T, -Q)
        eigvals = np.linalg.eigvals(P)
        if np.all(eigvals > 0):
            return P
    except:
        pass
    return np.eye(dim)

# --------------------------------------------------------------------------
# 2. SOS Methods (NO MULTIPROCESSING - Run in main process)
# --------------------------------------------------------------------------

def solve_sos_basic(dim, degree):
    """Basic SOS method."""
    print(f"  [Dim {dim}] Method: Basic SOS", flush=True)
    
    try:
        prog = MathematicalProgram()
        x_vars = prog.NewIndeterminates(dim, "x")
        
        M = get_dense_system_matrix(dim)
        f = M @ x_vars
        
        V_poly, V_gram = prog.NewSosPolynomial(Variables(x_vars), degree)
        V = V_poly.ToExpression()
        
        # V(0) = 0
        prog.AddLinearConstraint(V.Substitute({v: 0 for v in x_vars}) == 0)
        
        # Normalization
        sample_pt = {v: 0 for v in x_vars}
        sample_pt[x_vars[0]] = 0.1
        prog.AddLinearConstraint(V.Substitute(sample_pt) == 0.01)
        
        # Stability constraint
        dVdx = V.Jacobian(x_vars)
        V_dot = dVdx.dot(f)
        x_norm_sq = x_vars.dot(x_vars)
        epsilon = 1e-4
        
        prog.AddSosConstraint(Polynomial(-V_dot - epsilon * x_norm_sq))
        prog.AddCost(np.trace(V_gram))
        
        print(f"  [Dim {dim}] Solving...", flush=True)
        result = Solve(prog)
        
        if result.is_success():
            solver_name = result.get_solver_id().name()
            print(f"  [Dim {dim}] ✓ Success ({solver_name})", flush=True)
            return True, solver_name
        else:
            print(f"  [Dim {dim}] ✗ Infeasible", flush=True)
            return False, "INFEASIBLE"
            
    except Exception as e:
        print(f"  [Dim {dim}] ✗ Error: {e}", flush=True)
        return False, str(e)


def solve_sos_with_domain(dim, degree):
    """SOS with S-procedure domain constraint (Van der Pol style)."""
    print(f"  [Dim {dim}] Method: SOS with Domain Constraint", flush=True)
    
    try:
        prog = MathematicalProgram()
        x_vars = prog.NewIndeterminates(dim, "x")
        vars_obj = Variables(x_vars)
        
        M = get_dense_system_matrix(dim)
        f = M @ x_vars
        P = get_lyapunov_matrix(dim)
        
        # Define V
        V_poly, V_gram = prog.NewSosPolynomial(vars_obj, degree)
        V = V_poly.ToExpression()
        
        # V(0) = 0
        prog.AddLinearConstraint(V.Substitute({v: 0 for v in x_vars}) == 0)
        
        # Normalization
        sample_pt = {v: 0 for v in x_vars}
        sample_pt[x_vars[0]] = 0.1
        prog.AddLinearConstraint(V.Substitute(sample_pt) == 0.01)
        
        # Compute V_dot
        dVdx = V.Jacobian(x_vars)
        V_dot = dVdx.dot(f)
        
        # Domain constraint: ellipsoidal region
        gamma_domain = 10.0
        ellipse_expr = sum(P[i, j] * x_vars[i] * x_vars[j] 
                          for i in range(dim) for j in range(dim))
        boundary = gamma_domain - ellipse_expr
        
        # S-procedure with lambda multiplier
        lambda_degree = 2  # Keep lambda simple
        lambda_poly, _ = prog.NewSosPolynomial(vars_obj, lambda_degree)
        lambda_ = lambda_poly.ToExpression()
        
        # Key constraint: -V_dot - lambda * boundary is SOS
        expression = -V_dot - lambda_ * boundary
        prog.AddSosConstraint(Polynomial(expression.Expand(), vars_obj))
        prog.AddCost(np.trace(V_gram))
        
        print(f"  [Dim {dim}] Solving...", flush=True)
        result = Solve(prog)
        
        if result.is_success():
            solver_name = result.get_solver_id().name()
            print(f"  [Dim {dim}] ✓ Success ({solver_name})", flush=True)
            return True, solver_name
        else:
            print(f"  [Dim {dim}] ✗ Infeasible", flush=True)
            return False, "INFEASIBLE"
            
    except Exception as e:
        print(f"  [Dim {dim}] ✗ Error: {e}", flush=True)
        return False, str(e)


def solve_sos_relaxed(dim, degree):
    """Relaxed SOS with larger epsilon."""
    print(f"  [Dim {dim}] Method: Relaxed SOS", flush=True)
    
    try:
        prog = MathematicalProgram()
        x_vars = prog.NewIndeterminates(dim, "x")
        
        M = get_dense_system_matrix(dim)
        f = M @ x_vars
        
        V_poly, V_gram = prog.NewSosPolynomial(Variables(x_vars), degree)
        V = V_poly.ToExpression()
        
        prog.AddLinearConstraint(V.Substitute({v: 0 for v in x_vars}) == 0)
        
        sample_pt = {v: 0 for v in x_vars}
        sample_pt[x_vars[0]] = 0.1
        prog.AddLinearConstraint(V.Substitute(sample_pt) == 0.01)
        
        dVdx = V.Jacobian(x_vars)
        V_dot = dVdx.dot(f)
        x_norm_sq = x_vars.dot(x_vars)
        
        # Larger epsilon for relaxation
        epsilon = 1e-3
        
        prog.AddSosConstraint(Polynomial(-V_dot - epsilon * x_norm_sq))
        prog.AddCost(np.trace(V_gram))
        
        print(f"  [Dim {dim}] Solving...", flush=True)
        result = Solve(prog)
        
        if result.is_success():
            solver_name = result.get_solver_id().name()
            print(f"  [Dim {dim}] ✓ Success ({solver_name})", flush=True)
            return True, solver_name
        else:
            print(f"  [Dim {dim}] ✗ Infeasible", flush=True)
            return False, "INFEASIBLE"
            
    except Exception as e:
        print(f"  [Dim {dim}] ✗ Error: {e}", flush=True)
        return False, str(e)


def solve_sos_cascade(dim, degree):
    """Try multiple methods in sequence."""
    methods = [
        ("Basic", solve_sos_basic),
        ("Domain", solve_sos_with_domain),
        ("Relaxed", solve_sos_relaxed)
    ]
    
    for method_name, method_func in methods:
        success, info = method_func(dim, degree)
        if success:
            return True, f"{method_name}|{info}"
    
    return False, "ALL_FAILED"

# --------------------------------------------------------------------------
# 3. Runner (No multiprocessing, just timing)
# --------------------------------------------------------------------------

def run_sos_analysis(dim, method='cascade', degree=2, timeout=600):
    """Run SOS analysis with timeout."""
    print(f"\n{'='*70}")
    print(f"Dimension: {dim}, Method: {method.upper()}, Degree: {degree}")
    print(f"{'='*70}")
    
    method_map = {
        'basic': solve_sos_basic,
        'domain': solve_sos_with_domain,
        'relaxed': solve_sos_relaxed,
        'cascade': solve_sos_cascade
    }
    
    solver_func = method_map.get(method, solve_sos_cascade)
    
    # Simple timing (no multiprocessing)
    start_time = time.time()
    
    try:
        success, info = solver_func(dim, degree)
        duration = time.time() - start_time
        
        if success:
            print(f"\n✓ SUCCESS in {duration:.4f}s")
            print(f"  Details: {info}")
            return "SUCCESS", duration, info
        else:
            print(f"\n✗ FAILED in {duration:.4f}s")
            print(f"  Reason: {info}")
            return "FAILED", duration, info
            
    except KeyboardInterrupt:
        print(f"\n⚠ Interrupted by user")
        return "INTERRUPTED", time.time() - start_time, "User interrupt"
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n✗ ERROR in {duration:.4f}s")
        print(f"  Exception: {e}")
        return "ERROR", duration, str(e)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SOS Lyapunov Scaling Benchmark (No Multiprocessing)"
    )
    parser.add_argument('--method', 
                       choices=['basic', 'domain', 'relaxed', 'cascade'], 
                       default='cascade',
                       help='SOS method to use')
    parser.add_argument('--degree', type=int, default=2,
                       help='Polynomial degree')
    parser.add_argument('--dims', nargs='+', type=int, 
                       default=[2, 4, 6, 8, 10, 20, 50, 100],
                       help='Dimensions to test')
    args = parser.parse_args()
    
    print(f"\n{'#'*70}")
    print(f"# SCALING BENCHMARK: DENSE ROTATED OSCILLATOR")
    print(f"# Method: {args.method.upper()}, Degree: {args.degree}")
    print(f"{'#'*70}\n")
    
    results = []
    
    for n in args.dims:
        status, duration, info = run_sos_analysis(n, args.method, args.degree)
        results.append((n, status, duration, info))
        
        # Stop on failures for cascade mode
        if args.method == 'cascade' and status != 'SUCCESS':
            print("\n⚠ Stopping benchmark due to failure in cascade mode.")
            break
    
    # Final summary
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dim':<6} | {'Status':<12} | {'Time (s)':<12} | {'Details':<30}")
    print("-" * 70)
    
    for dim, status, duration, info in results:
        info_short = info[:28] + ".." if len(info) > 30 else info
        print(f"{dim:<6} | {status:<12} | {duration:<12.4f} | {info_short:<30}")
    
    successes = sum(1 for _, s, _, _ in results if s == 'SUCCESS')
    total = len(results)
    
    print(f"\n{'='*70}")
    print(f"Success Rate: {successes}/{total} ({100*successes/total:.1f}%)")
    
    if successes > 0:
        success_times = [d for _, s, d, _ in results if s == 'SUCCESS']
        print(f"Average time (successful): {np.mean(success_times):.2f}s")
        print(f"Max time (successful): {np.max(success_times):.2f}s")
        print(f"Min time (successful): {np.min(success_times):.2f}s")
    print(f"{'='*70}\n")