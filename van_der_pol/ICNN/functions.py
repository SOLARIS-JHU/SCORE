from dreal import *
import torch
import numpy as np
import random

def build_dreal_icnn(model, vars_):
    """
    Converts the CertifiableICNN into a dReal symbolic expression.
    
    Returns:
        V_sym: The symbolic expression for the Lyapunov output.
        grad_sym: List of symbolic expressions for the gradient [dV/dx1, dV/dx2, ...]
                  (Explicit gradient construction is much faster for SMT than auto-diff).
    """
    # 1. Extract Parameters as numpy arrays
    W_list = [w.detach().numpy() for w in model.W]
    b_list = [b.detach().numpy() for b in model.bias]
    
    # softplus(x) = log(1 + exp(x))
    U_list = [np.log(1 + np.exp(u.detach().numpy())) for u in model.U]

    # 2. Setup Inputs and Gradients (Identity matrix for dx/dx)
    # x_val is the list of symbolic variables
    # x_grad is the identity matrix (Jacobian of x with respect to itself)
    dx = [
        [1.0 if i == j else 0.0 for j in range(len(vars_))] 
        for i in range(len(vars_))
    ]
    
    x_col = np.array(vars_) 
    dx_col = np.array(dx)   

    # --- Helper Functions for Symbolic Math ---
    def symbolic_linear(W, b, x_val, x_grad):
        # z = Wx + b
        z_val = np.dot(W, x_val) + b.reshape(-1)
        # dz = W * dx
        z_grad = np.dot(W, x_grad)
        return z_val, z_grad

    def symbolic_softplus(z_val, z_grad):
        out_val = []
        out_grad = []
        for i in range(len(z_val)):
            val = z_val[i]
            grad = z_grad[i]
            
            # Symbolic Softplus: log(1 + exp(x))
            s_val = log(1.0 + exp(val))
            out_val.append(s_val)
            
            # Symbolic Derivative of Softplus is Sigmoid: 1 / (1 + exp(-x))
            sigmoid = 1.0 / (1.0 + exp(-val))
            out_grad.append([sigmoid * g for g in grad])
            
        return np.array(out_val), np.array(out_grad)

    # --- Layer 0 ---
    # z = Softplus(W[0]x + b[0])
    u0_val, u0_grad = symbolic_linear(W_list[0], b_list[0], x_col, dx_col)
    z_val, z_grad = symbolic_softplus(u0_val, u0_grad)

    # --- Hidden Layers ---
    # z_{k+1} = Softplus(W[k+1]x + b[k+1] + U[k] * z_k)
    # Note: U_list corresponds to the transformations between hidden layers
    for k in range(len(U_list) - 1):
        U, W, b = U_list[k], W_list[k+1], b_list[k+1]
        
        lin_val, lin_grad = symbolic_linear(W, b, x_col, dx_col)
        cvx_val = np.dot(U, z_val)
        cvx_grad = np.dot(U, z_grad)
        
        z_val, z_grad = symbolic_softplus(lin_val + cvx_val, lin_grad + cvx_grad)

    # --- Output Layer ---
    # out = W[last]x + b[last] + U[last] * z_last
    W_last, b_last, U_last = W_list[-1], b_list[-1], U_list[-1]
    
    lin_val, lin_grad = symbolic_linear(W_last, b_last, x_col, dx_col)
    cvx_val = np.dot(U_last, z_val)
    cvx_grad = np.dot(U_last, z_grad)
    
    nn_out = (lin_val + cvx_val)[0]
    nn_grad = (lin_grad + cvx_grad)[0]

    # --- Correction for V(0) = 0 ---
    # We must subtract the network output at x=0 to ensure V(0)=0.
    # We calculate this concretely to avoid complex symbolic expression at 0.
    zeros_in = torch.zeros(1, len(vars_))
    # We need a quick forward pass of the model to get y_zero
    with torch.no_grad():
        y_zero = model._icnn_pass(zeros_in).item()
        
    final_V = nn_out - y_zero
    
    return final_V, nn_grad

def CheckLyapunov(x, f, V, grad_V, ball_lb, ball_ub, config, epsilon):    
    """
    Checks if V violates Lyapunov conditions
    
    Inputs:
        x: List of dReal variables
        f: List of expressions for dynamics dx/dt
        V: Expression for V(x)
        grad_V: List of expressions for dV/dx (Explicitly provided for speed)
    """
    ball = Expression(0)
    lie_derivative_of_V = Expression(0)
    
    for i in range(len(x)):
        ball += x[i]*x[i]
        # Instead of V.Differentiate(x[i]), we use the explicit gradient constructed in build_icnn
        lie_derivative_of_V += f[i] * grad_V[i]  
        
    # Domain: ball_lb^2 <= ||x||^2 <= ball_ub^2
    ball_in_bound = logical_and(ball_lb*ball_lb <= ball, ball <= ball_ub*ball_ub)
    
    # We want to verify: Domain => (V > 0  AND  Lie_V <= -epsilon)
    # We check the negation (Counter Example Search): Domain AND (V <= 0  OR  Lie_V > -epsilon)
    
    condition = logical_and(
        logical_imply(ball_in_bound, V >= 0),
        logical_imply(ball_in_bound, lie_derivative_of_V <= -epsilon)
    )
    
    # Check satisfiability of the negation
    return CheckSatisfiability(logical_not(condition), config)

def AddCounterexamples(x, CE, N): 
    """
    Parses the dReal Counter-Example (Box) and adds N random points 
    from that region to the training set x.
    """
    if not CE:
        return x
        
    # Assuming x is [Batch, Dim]
    dim = x.shape[1]
    
    # dReal returns a dictionary of Variable -> Interval
    # We sort keys by name to ensure x1 maps to dim 0, x2 to dim 1, etc.
    keys = sorted([k for k in CE.keys()], key=lambda v: str(v))
    
    if len(keys) < dim:
        return x

    intervals = []
    for k in keys:
        intervals.append((CE[k].lb(), CE[k].ub()))
        
    new_points = []
    for _ in range(N):
        pt = []
        for i in range(dim):
            lb, ub = intervals[i]
            pt.append(random.uniform(lb, ub))
        new_points.append(pt)
        
    new_tensor = torch.tensor(new_points, dtype=torch.float32)
    return torch.cat((x, new_tensor), 0)