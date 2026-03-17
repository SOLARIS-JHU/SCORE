from dreal import *
import torch
import numpy as np
import random

def build_dreal_lyapunov(model, vars_):
    """
    Converts a PyTorch Neural Network into a dReal symbolic expression.
    """
    exprs = vars_
    
    # Iterate through PyTorch layers to construct the symbolic graph
    for layer in model.children():
        if isinstance(layer, torch.nn.Linear):
            weight = layer.weight.detach().numpy()
            bias = layer.bias.detach().numpy() if layer.bias is not None else np.zeros(layer.out_features)
            new_exprs = []
            for i in range(weight.shape[0]):
                e = float(bias[i])
                for j in range(weight.shape[1]):
                    e += float(weight[i,j]) * exprs[j]
                new_exprs.append(e)
            exprs = new_exprs
        elif isinstance(layer, torch.nn.Tanh):
            exprs = [tanh(e) for e in exprs]
        elif isinstance(layer, torch.nn.ReLU):
            # Note: ReLU is hard for SMT solvers, Tanh is preferred for smooth Lyapunov functions
            # But if used, we can represent ReLU(x) as max(0, x)
            exprs = [max_(0.0, e) for e in exprs]
            
    return exprs[0] # Assuming single output V

def CheckLyapunov(x, f, V, ball_lb, ball_ub, config, epsilon):    
    """
    Checks if V violates Lyapunov conditions:
    1. V(x) > 0
    2. Lie_V(x) < -epsilon
    Returns a counter-example (Box) if violations exist, otherwise None.
    """
    ball = Expression(0)
    lie_derivative_of_V = Expression(0)
    
    for i in range(len(x)):
        ball += x[i]*x[i]
        lie_derivative_of_V += f[i]*V.Differentiate(x[i])  
        
    # Domain: ball_lb^2 <= ||x||^2 <= ball_ub^2
    ball_in_bound = logical_and(ball_lb*ball_lb <= ball, ball <= ball_ub*ball_ub)
    
    # We want to verify: Domain => (V > 0  AND  Lie_V <= -epsilon)
    # We check the negation: Domain AND (V <= 0  OR  Lie_V > -epsilon)
    
    # Note on epsilon: For strict stability, Lie_V should be negative. 
    # We look for points where Lie_V is "not negative enough" (i.e., > -epsilon).
    
    condition = logical_and(logical_imply(ball_in_bound, V >= 0),
                            logical_imply(ball_in_bound, lie_derivative_of_V <= -epsilon))
    
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
    # We need to map these back to the dimensions of x.
    # We sort keys by name to ensure x1 maps to dim 0, x2 to dim 1, etc.
    keys = sorted([k for k in CE.keys()], key=lambda v: str(v))
    
    if len(keys) < dim:
        # Fallback if CE doesn't contain all vars (e.g. if one var was optimized out)
        return x

    intervals = []
    for k in keys:
        intervals.append((CE[k].lb(), CE[k].ub()))
        
    new_points = []
    for _ in range(N):
        pt = []
        for i in range(dim):
            lb, ub = intervals[i]
            # Add small noise to avoid sampling exactly on the boundary repeatedly
            pt.append(random.uniform(lb, ub))
        new_points.append(pt)
        
    new_tensor = torch.tensor(new_points, dtype=torch.float32)
    return torch.cat((x, new_tensor), 0)