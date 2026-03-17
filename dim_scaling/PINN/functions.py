from dreal import *
import torch
import torch.nn as nn
import numpy as np

def build_dreal_network(model, vars_):
    """Recursively builds dReal expression from PyTorch Sequential model."""
    exprs = [Expression(v) for v in vars_]
    for layer in model:
        if isinstance(layer, nn.Linear):
            weight = layer.weight.detach().cpu().numpy()
            bias = layer.bias.detach().cpu().numpy() if layer.bias is not None else np.zeros(layer.out_features)
            new_exprs = []
            for i in range(weight.shape[0]):
                e = float(bias[i])
                for j in range(weight.shape[1]):
                    e += float(weight[i,j]) * exprs[j]
                new_exprs.append(e)
            exprs = new_exprs
        elif isinstance(layer, nn.Tanh):
            exprs = [tanh(e) for e in exprs]
        elif isinstance(layer, nn.Sigmoid):
            exprs = [1.0 / (1.0 + exp(-e)) for e in exprs]
    return exprs[0]

def CheckZubovScaling(vars_, f_sym, V_sym, level_c, ball_ub, config, epsilon=1e-4):
    """N-Dimensional Zubov Verification"""
    # 1. Geometry: Norm Squared
    x_norm_sq = sum([v*v for v in vars_])
    
    # 2. Lie Derivative & Radial Check
    grad_V = [V_sym.Differentiate(v) for v in vars_]
    lie_deriv = sum([f_sym[i] * grad_V[i] for i in range(len(vars_))])
    radial_deriv = sum([vars_[i] * grad_V[i] for i in range(len(vars_))])

    # Domain: ||x|| <= R
    in_ball = (x_norm_sq <= ball_ub**2)
    
    # Condition: (V <= c) => (Lie < 0)
    # Violation: (V <= c) AND (Lie >= -epsilon)
    violation_lie = logical_and(V_sym <= level_c, lie_deriv >= -epsilon)
    
    # Condition: (V near c) => (V increasing radially)
    violation_geom = logical_and(V_sym >= level_c - 0.05, radial_deriv <= 0)

    total_violation = logical_and(in_ball, logical_or(violation_lie, violation_geom))
    return CheckSatisfiability(total_violation, config)