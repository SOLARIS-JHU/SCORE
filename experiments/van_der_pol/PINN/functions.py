from dreal import *
import torch
import torch.nn as nn
import numpy as np

def build_dreal_layer(module, inputs):
    """Recursively builds dReal expression from PyTorch module."""
    if isinstance(module, nn.Sequential):
        for layer in module:
            inputs = build_dreal_layer(layer, inputs)
        return inputs
    elif isinstance(module, nn.Linear):
        weight = module.weight.detach().cpu().numpy()
        bias = module.bias.detach().cpu().numpy() if module.bias is not None else np.zeros(module.out_features)
        new_exprs = []
        for i in range(weight.shape[0]):
            e = float(bias[i])
            for j in range(weight.shape[1]):
                e += float(weight[i,j]) * inputs[j]
            new_exprs.append(e)
        return new_exprs
    elif isinstance(module, nn.Tanh):
        return [tanh(e) for e in inputs]
    elif isinstance(module, nn.Sigmoid):
        return [1.0 / (1.0 + exp(-e)) for e in inputs]
    
    # Handle nested modules
    children = list(module.children())
    if len(children) > 0:
        for child in children:
            inputs = build_dreal_layer(child, inputs)
        return inputs
    return inputs

def build_dreal_network(model, vars_):
    exprs = [Expression(v) for v in vars_]
    outputs = build_dreal_layer(model, exprs)
    return outputs[0]

def CheckZubovLevelSet(vars_, f_sym, V_sym, level_c, ball_lb, ball_ub, config, epsilon=1e-5):
    """
    Verifies V(x) <= c is an invariant ROA.
    Includes a RADIAL GRADIENT CHECK to prevent 'leaking' past the limit cycle.
    """
    if not isinstance(V_sym, Expression):
        V_sym = Expression(V_sym)
    
    # 1. Geometry Setup
    x_norm_sq = vars_[0]*vars_[0] + vars_[1]*vars_[1]
    
    # 2. Compute Gradients Symbolic
    grad_V = [V_sym.Differentiate(v) for v in vars_]
    
    # 3. Lie Derivative: sum( f_i * dV/dx_i )
    lie_deriv = Expression(0)
    for i in range(len(vars_)):
        lie_deriv += f_sym[i] * grad_V[i]
        
    # 4. Radial Alignment: dot(x, grad_V)
    # On the boundary, V must be increasing radially.
    # If V starts decreasing (outer slope), this becomes negative.
    radial_deriv = Expression(0)
    for i in range(len(vars_)):
        radial_deriv += vars_[i] * grad_V[i]

    # --- Constraints ---
    
    # A. Domain Ring
    in_ball = logical_and(x_norm_sq >= ball_lb**2, x_norm_sq <= ball_ub**2)
    
    # B. Lie Derivative Check (Standard Invariance)
    # Condition: (V <= c) AND (in_ball) => (Lie < 0)
    # Violation: (V <= c) AND (in_ball) AND (Lie >= -epsilon)
    violation_lie = logical_and(V_sym <= level_c, lie_deriv >= -epsilon)
    
    # C. Geometry Check (Volcano Prevention)
    # Condition: (V ~= c) => (Radial_Deriv > 0)
    # We check a thin shell near the boundary [c-delta, c]
    # Violation: (V >= c - 0.05) AND (V <= c) AND (Radial <= 0)
    violation_geom = logical_and(V_sym >= level_c - 0.05, radial_deriv <= 0)

    # Combine: Fail if EITHER Lyapunov is wrong OR Geometry is wrong
    # Note: We check violation inside the ball and inside the level set
    total_violation = logical_and(in_ball, logical_and(V_sym <= level_c, logical_or(violation_lie, violation_geom)))
    
    return CheckSatisfiability(total_violation, config)


def FindMaxROA(model, f_dreal_func, lb, ub, step=0.05):
    """
    Bisection search using the rigorous CheckZubovLevelSet.
    """
    config = Config()
    config.use_polytope = True
    config.precision = 1e-4 
    
    x1 = Variable("x1")
    x2 = Variable("x2")
    vars_ = [x1, x2]
    
    f_sym = f_dreal_func(vars_) 

    model.cpu()
    V_sym = build_dreal_network(model, vars_)
    
    print("\n--- Starting dReal Verification (w/ Radial Check) ---")
    verified_c = 0.0
    
    # Bisection Parameters
    low = 0.1
    high = 0.99
    tolerance = 0.01
    
    while (high - low) > tolerance:
        mid = (low + high) / 2.0
        print(f"Verifying c={mid:.4f}...", end="", flush=True)
        
        # Check
        result = CheckZubovLevelSet(vars_, f_sym, V_sym, mid, lb, ub, config)
        
        if not result:
            print(" [SAFE]")
            verified_c = mid
            low = mid # Try to go higher
        else:
            print(" [UNSAFE]")
            high = mid # Too high, go lower
            
    print(f"Max Verified Level Set: {verified_c:.4f}")
    return verified_c