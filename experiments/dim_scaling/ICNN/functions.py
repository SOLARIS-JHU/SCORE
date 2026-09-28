from dreal import *
import torch
import numpy as np
import random

def build_dreal_icnn(model, vars_):
    """
    Constructs ICNN symbolic expression AND its gradient.
    Uses Softplus (smooth) to avoid combinatorial explosion.
    """
    # 1. Setup Inputs and Gradients
    dx = [
        [1.0 if i == j else 0.0 for j in range(len(vars_))] 
        for i in range(len(vars_))
    ]
    
    W_list = [w.detach().numpy() for w in model.W]
    b_list = [b.detach().numpy() for b in model.bias]
    U_list = [np.log(1 + np.exp(u.detach().numpy())) for u in model.U]

    def symbolic_linear(W, b, x_val, x_grad):
        z_val = np.dot(W, x_val) + b.reshape(-1)
        z_grad = np.dot(W, x_grad)
        return z_val, z_grad

    def symbolic_softplus(z_val, z_grad):
        out_val = []
        out_grad = []
        for i in range(len(z_val)):
            val = z_val[i]
            grad = z_grad[i]
            # Softplus: log(1 + exp(x))
            s_val = log(1.0 + exp(val))
            out_val.append(s_val)
            # Sigmoid gradient: 1 / (1 + exp(-x))
            sigmoid = 1.0 / (1.0 + exp(-val))
            out_grad.append([sigmoid * g for g in grad])
        return np.array(out_val), np.array(out_grad)

    # 2. First Layer
    x_col = np.array(vars_) 
    dx_col = np.array(dx)   
    u0_val, u0_grad = symbolic_linear(W_list[0], b_list[0], x_col, dx_col)
    z_val, z_grad = symbolic_softplus(u0_val, u0_grad)

    # 3. Hidden Layers
    for k in range(len(U_list) - 1):
        U, W, b = U_list[k], W_list[k+1], b_list[k+1]
        lin_val, lin_grad = symbolic_linear(W, b, x_col, dx_col)
        cvx_val = np.dot(U, z_val)
        cvx_grad = np.dot(U, z_grad)
        z_val, z_grad = symbolic_softplus(lin_val + cvx_val, lin_grad + cvx_grad)

    # 4. Output Layer
    W_last, b_last, U_last = W_list[-1], b_list[-1], U_list[-1]
    lin_val, lin_grad = symbolic_linear(W_last, b_last, x_col, dx_col)
    cvx_val = np.dot(U_last, z_val)
    cvx_grad = np.dot(U_last, z_grad)
    
    return (lin_val + cvx_val)[0], (lin_grad + cvx_grad)[0]

def CheckLyapunov(x, f, V, grad_V, ball_lb, ball_ub, config, epsilon, timeout=30):    
    """
    Synchronous Verification.
    Returns:
        None -> Verified (UNSAT)
        Box  -> Counter-example found (SAT)
    """
    ball = Expression(0)
    lie_derivative = Expression(0)
    
    for i in range(len(x)):
        ball += x[i]*x[i]
        lie_derivative += f[i] * grad_V[i]
        
    ball_in_bound = logical_and(ball_lb*ball_lb <= ball, ball <= ball_ub*ball_ub)
    
    # 1. Check Positivity Violation (V <= 0)
    # We want V > 0. We look for V <= 0.
    query_pos = logical_and(ball_in_bound, V <= 0)
    result_pos = CheckSatisfiability(query_pos, config)
    if result_pos: return result_pos 

    # 2. Check Derivative Violation (Lie_V >= -epsilon)
    # We want Lie_V < -epsilon. We look for Lie_V >= -epsilon.
    query_der = logical_and(ball_in_bound, lie_derivative >= -epsilon)
    result_der = CheckSatisfiability(query_der, config)
    if result_der: return result_der 

    return None # Verified

def AddCounterexamples(x, CE, N): 
    if not CE: return x
    dim = x.shape[1]
    keys = sorted([k for k in CE.keys()], key=lambda v: str(v))
    intervals = [(CE[k].lb(), CE[k].ub()) for k in keys]
    
    new_points = []
    margin = 0.1 
    for _ in range(N):
        pt = [random.uniform(i[0]-margin, i[1]+margin) for i in intervals]
        new_points.append(pt)
    return torch.cat((x, torch.tensor(new_points, dtype=torch.float32)), 0)