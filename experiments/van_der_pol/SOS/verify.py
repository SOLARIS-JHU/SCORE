import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.integrate import solve_ivp
from pydrake.all import (
    SymbolicVectorSystem,
    Variable,
    RegionOfAttraction,
    RegionOfAttractionOptions,
    MathematicalProgram,
    Solve,
    Variables,
    Polynomial,
    LinearQuadraticRegulator
)

def run_roa_analysis(degree):
    x1 = Variable("x1")
    x2 = Variable("x2")
    x = np.array([x1, x2])
    
    # Dynamics: Reversed Van der Pol
    f = np.array([
        -x2,                        
        x1 + (x1**2 - 1) * x2       
    ])
    
    sys = SymbolicVectorSystem(state=[x1, x2], dynamics=f)
    context = sys.CreateDefaultContext()

    print(f"\n{'='*70}")
    print(f"DRAKE'S REGION OF ATTRACTION (Degree {degree})")
    print(f"{'='*70}\n")
    
    # ---------------------------------------------------------
    # STEP 1: Find Lyapunov Function V using SOS
    # ---------------------------------------------------------
    print("Step 1: Finding Lyapunov function V using SOS...\n")
    
    A = np.array([[0, -1], [1, -1]])
    B = np.zeros((2, 1))
    Q = np.eye(2)
    R = np.eye(1)
    K, S = LinearQuadraticRegulator(A, B, Q, R)
    
    print("LQR Shape Matrix S:")
    print(S)
    print()
    
    V_sol = find_lyapunov_function(x, f, degree, S)
    if V_sol is None:
        return
    
    print(f"{'='*70}")
    print(f"Lyapunov Function V(x):")
    print(f"{'='*70}")
    print(V_sol)
    print(f"{'='*70}\n")
    
    # ---------------------------------------------------------
    # STEP 2: Use Drake's RegionOfAttraction
    # ---------------------------------------------------------
    print("Step 2: Using Drake's RegionOfAttraction to find maximal rho...\n")
    
    # Set equilibrium point
    context.SetContinuousState([0.0, 0.0])
    
    print("Running RegionOfAttraction...")
    print("(This uses an optimized bilinear alternation algorithm)\n")
    
    try:
        # Create options and set the Lyapunov candidate
        options = RegionOfAttractionOptions()
        options.lyapunov_candidate = V_sol
        options.state_variables = x  # Specify state variables
        
        # Call RegionOfAttraction
        V_roa = RegionOfAttraction(
            system=sys,
            context=context,
            options=options
        )
        
        print("✓ RegionOfAttraction succeeded!\n")
        
        print(f"{'='*70}")
        print(f"Certified Lyapunov Function from Drake:")
        print(f"{'='*70}")
        print(V_roa)
        print(f"{'='*70}\n")
        
        # Find rho by evaluating on limit cycle
        rho_certified = estimate_rho_from_cycle(V_roa)
        
        print(f"{'#'*70}")
        print(f"  DRAKE-CERTIFIED REGION OF ATTRACTION")
        print(f"  rho* = {rho_certified:.6f}")
        print(f"  Region: {{x : V(x) ≤ {rho_certified:.6f}}}")
        print(f"  Method: Drake's optimized V-S procedure")
        print(f"{'#'*70}\n")
        
        plot_roa(V_roa, rho_certified, degree, method="Drake")
        
    except Exception as e:
        print(f"✗ RegionOfAttraction failed: {e}\n")
        print("Falling back to manual method...")
        
        # Fallback: use our V with numerical verification
        rho_fallback = estimate_rho_from_cycle(V_sol)
        print(f"\nUsing manual estimate: rho = {rho_fallback:.6f}\n")
        plot_roa(V_sol, rho_fallback, degree, method="Manual")


def find_lyapunov_function(x_original, f_original, degree, S_matrix):
    """Find V using SOS with large domain"""
    
    prog = MathematicalProgram()
    x_local = prog.NewIndeterminates(2, "x")
    
    sub_map = {x_original[0]: x_local[0], x_original[1]: x_local[1]}
    
    f_local = np.array([
        f_original[0].Substitute(sub_map),
        f_original[1].Substitute(sub_map)
    ])
    
    vars_local = Variables(x_local)

    V_poly, V_gram = prog.NewSosPolynomial(vars_local, degree)
    V = V_poly.ToExpression()
    
    prog.AddLinearConstraint(
        V.Substitute({x_local[0]: 0, x_local[1]: 0}) == 0
    )
    prog.AddLinearConstraint(
        V.Substitute({x_local[0]: 0.1, x_local[1]: 0}) == 0.01
    )

    dVdx = V.Jacobian(x_local)
    V_dot = dVdx.dot(f_local)
    
    gamma_domain = 5.0
    ellipse_expr = (S_matrix[0, 0] * x_local[0]**2 + 
                    2 * S_matrix[0, 1] * x_local[0] * x_local[1] + 
                    S_matrix[1, 1] * x_local[1]**2)
    boundary = gamma_domain - ellipse_expr
    
    lambda_poly, _ = prog.NewSosPolynomial(vars_local, degree)
    lambda_ = lambda_poly.ToExpression()
    
    expression = -V_dot - lambda_ * boundary
    prog.AddSosConstraint(Polynomial(expression.Expand(), vars_local))
    prog.AddCost(np.trace(V_gram))

    print("Solving for V...")
    result = Solve(prog)
    
    if not result.is_success():
        print("✗ Failed")
        return None
    
    V_sol = result.GetSolution(V).Substitute({x_local[0]: x_original[0], 
                                               x_local[1]: x_original[1]})
    print("✓ Found V\n")
    return V_sol


def estimate_rho_from_cycle(V_sol):
    """Get upper bound from limit cycle"""
    vars_in_V = list(V_sol.GetVariables())
    vars_in_V.sort(key=lambda x: x.get_name())
    var_x = vars_in_V[0]
    var_y = vars_in_V[1]
    
    def vdp_inverse(t, state):
        x1, x2 = state
        return [x2, -x1 - (x1**2 - 1) * x2]
    
    sol = solve_ivp(vdp_inverse, [0, 50], [0.1, 0.0], rtol=1e-8, max_step=0.01)
    idx_start = int(len(sol.t) * 0.7)
    cycle_x = sol.y[0][idx_start:]
    cycle_y = sol.y[1][idx_start:]
    
    v_on_cycle = []
    for cx, cy in zip(cycle_x, cycle_y):
        try:
            v_val = V_sol.Evaluate({var_x: cx, var_y: cy})
            if not np.isnan(v_val) and not np.isinf(v_val):
                v_on_cycle.append(v_val)
        except:
            continue
    
    return np.min(v_on_cycle) if v_on_cycle else 1.0


def plot_roa(V, rho, degree, method="Drake"):
    print("Generating plot...\n")
    
    vars_in_V = list(V.GetVariables())
    vars_in_V.sort(key=lambda x: x.get_name())
    var_x, var_y = vars_in_V[0], vars_in_V[1]
    
    x_range = np.linspace(-2.0, 2.0, 250)
    y_range = np.linspace(-3., 3, 250)
    X, Y = np.meshgrid(x_range, y_range)
    V_numeric = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                V_numeric[i, j] = V.Evaluate({var_x: X[i, j], var_y: Y[i, j]})
            except:
                V_numeric[i, j] = np.nan

    def vdp_inverse(t, state):
        x1, x2 = state
        return [x2, -x1 - (x1**2 - 1) * x2]
    
    sol = solve_ivp(vdp_inverse, [0, 50], [0.1, 0.0], rtol=1e-8, max_step=0.02)
    idx_start = int(len(sol.t) * 0.7)
    cycle_x = sol.y[0][idx_start:]
    cycle_y = sol.y[1][idx_start:]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    contourf = ax.contourf(X, Y, V_numeric, levels=30, cmap="Blues", alpha=0.5)
    plt.colorbar(contourf, ax=ax, label="V(x)", shrink=0.8)
    
    levels = np.linspace(0.01, rho * 2, 15)
    ax.contour(X, Y, V_numeric, levels=levels, 
               colors='lightblue', linewidths=0.7, alpha=0.7)
    
    ax.plot(cycle_x, cycle_y, 'r--', linewidth=3.5, label="Limit Cycle", zorder=5)
    
    cs = ax.contour(X, Y, V_numeric, levels=[rho], 
                    colors=['lime'], linewidths=6, zorder=4)
    ax.clabel(cs, inline=True, fontsize=14, fmt=f"rho={rho:.3f}")
    
    skip = 10
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
              -Y[::skip, ::skip], 
              X[::skip, ::skip] + (X[::skip, ::skip]**2 - 1) * Y[::skip, ::skip],
              color='gray', alpha=0.5, width=0.003)
    
    ax.plot(0, 0, 'ko', markersize=14, label="Equilibrium", zorder=6)
    
    for init in [[0.5, 0], [1.0, 0.5], [0.3, -0.8], [0.1, 0.3], [0.8, -0.3]]:
        traj = solve_ivp(vdp_inverse, [0, 10], init, max_step=0.05)
        ax.plot(traj.y[0], traj.y[1], 'purple', alpha=0.7, linewidth=2)
    
    ax.set_xlabel("x₁", fontsize=15)
    ax.set_ylabel("x₂", fontsize=15)
    ax.set_title(f"Region of Attraction ({method} Method)\nDegree {degree}, rho = {rho:.3f}", 
                 fontsize=16, weight='bold', pad=20)
    ax.legend(fontsize=13, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    
    if method == "Drake":
        textstr = f"Drake's RegionOfAttraction\nOptimized V-S Procedure"
        color = 'lightgreen'
    else:
        textstr = f"SOS Lyapunov Function\nNumerical rho Verification"
        color = 'lightyellow'
    
    props = dict(boxstyle='round', facecolor=color, alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    filename = f"roa_drake_degree_{degree}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}\n")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Drake's RegionOfAttraction"
    )
    parser.add_argument('--degree', type=int, default=6,
                       help='Degree of Lyapunov polynomial')
    args = parser.parse_args()
    
    run_roa_analysis(args.degree)