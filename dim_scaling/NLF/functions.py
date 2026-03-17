from dreal import *
import torch
import numpy as np
import random
import multiprocessing

def build_dreal_lyapunov(model, vars_):
    """
    Converts a PyTorch Neural Network into a dReal symbolic expression.
    """
    exprs = vars_
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
            exprs = [max_(0.0, e) for e in exprs]
            
    return exprs[0]

def _check_sat_process(condition, config, result_queue):
    """Worker function for timeout handling"""
    result = CheckSatisfiability(logical_not(condition), config)
    result_queue.put(result)

def CheckLyapunov(x, f, V, ball_lb, ball_ub, config, epsilon, timeout=30):    
    """
    Checks V violations with a strict Timeout.
    """
    ball = Expression(0)
    lie_derivative_of_V = Expression(0)
    
    for i in range(len(x)):
        ball += x[i]*x[i]
        lie_derivative_of_V += f[i]*V.Differentiate(x[i])  
        
    ball_in_bound = logical_and(ball_lb*ball_lb <= ball, ball <= ball_ub*ball_ub)
    
    # Check: Domain => (V > 0 AND Lie_V <= -epsilon)
    condition = logical_and(logical_imply(ball_in_bound, V >= 0),
                            logical_imply(ball_in_bound, lie_derivative_of_V <= -epsilon))
    
    # Run dReal in a separate process to enforce timeout
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_check_sat_process, args=(condition, config, queue))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        print(f"  [TIMEOUT] dReal took longer than {timeout}s")
        return "TIMEOUT"
    
    if not queue.empty():
        return queue.get()
    return None

def AddCounterexamples(x, CE, N): 
    if not CE or CE == "TIMEOUT":
        return x
        
    dim = x.shape[1]
    # Map variable names (e.g., "x_0", "x_1") back to indices
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