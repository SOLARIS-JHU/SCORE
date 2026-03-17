import torch
import torch.nn as nn
import numpy as np
import time
import multiprocessing
import random
from dreal import *

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
MASTER_SEED = 42
TIMEOUT_SEC = 600       # 10 minutes timeout

# --------------------------------------------------------------------------
# 1. Scalable DENSE Dynamics (Rotated Damped Oscillator)
# --------------------------------------------------------------------------

def get_dense_system_matrix(dim):
    """
    Generates a dense, stable system matrix M = R * A * R^T.
    """
    # 1. Base System (Sparse) - Block diagonal: [[0, 1], [-1, -0.5]]
    A = np.zeros((dim, dim))
    for i in range(0, dim, 2):
        A[i, i+1] = 1.0
        A[i+1, i] = -1.0
        A[i+1, i+1] = -0.5
        
    # 2. Random Rotation (Dense mixing)
    # Use MASTER_SEED + dim to ensure reproducibility per dimension
    np.random.seed(MASTER_SEED + dim) 
    
    H = np.random.randn(dim, dim)
    Q, _ = np.linalg.qr(H)
    M = Q @ A @ Q.T
    return torch.tensor(M, dtype=torch.float32)

# Global storage for the current matrix
CURRENT_MATRIX = None 

def f_value(x):
    """ Dense Dynamics: dx = x * M^T """
    global CURRENT_MATRIX
    return x @ CURRENT_MATRIX.t()

def f_dreal(vars_):
    """ Dense Dynamics for dReal """
    global CURRENT_MATRIX
    dim = len(vars_)
    M_np = CURRENT_MATRIX.numpy()
    dx = []
    for i in range(dim):
        val = 0.0
        for j in range(dim):
            val += float(M_np[i, j]) * vars_[j]
        dx.append(val)
    return dx

# --------------------------------------------------------------------------
# 2. dReal Helper Functions (Zubov Specific)
# --------------------------------------------------------------------------

def build_dreal_layer(module, inputs):
    if isinstance(module, nn.Sequential):
        for layer in module:
            inputs = build_dreal_layer(layer, inputs)
        return inputs
    elif isinstance(module, nn.Linear):
        weight = module.weight.detach().numpy()
        bias = module.bias.detach().numpy() if module.bias is not None else np.zeros(module.out_features)
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
    return inputs

def _check_zubov_process(condition, config, result_queue):
    """Worker for timeout handling"""
    result = CheckSatisfiability(logical_not(condition), config)
    result_queue.put(result)

def CheckZubovCondition(vars_, f_sym, V_sym, ball_lb, ball_ub, config, timeout=30):
    """
    Verifies the Zubov PDE condition with a strict timeout.
    """
    # 1. Geometry
    x_norm_sq = Expression(0)
    for v in vars_: x_norm_sq += v*v
    
    # 2. Gradients
    grad_V = [V_sym.Differentiate(v) for v in vars_]
    
    # 3. Lie Derivative
    lie_deriv = Expression(0)
    for i in range(len(vars_)):
        lie_deriv += f_sym[i] * grad_V[i]
        
    # 4. Phi (forcing function)
    # Using simple quadratic phi(x) = 0.1 * ||x||^2
    phi = 0.1 * x_norm_sq
    
    # 5. Zubov Error
    # Theoretical: Lie = -phi * (1 - V)
    # We check if Lie > -phi * (1 - V) + tolerance (i.e. not decaying fast enough)
    epsilon = 0.05
    target = -phi * (1.0 - V_sym)
    
    # Domain: Don't check at 0 (singularity), check in ring
    in_domain = logical_and(x_norm_sq >= ball_lb**2, x_norm_sq <= ball_ub**2)
    
    # We want to prove: In_Domain => (Lie <= Target + eps)
    # Counter-example:  In_Domain AND (Lie > Target + eps)
    violation = logical_and(in_domain, lie_deriv > target + epsilon)
    
    # Run with Timeout
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_check_zubov_process, args=(logical_not(violation), config, queue))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return "TIMEOUT"
    
    if not queue.empty():
        # If queue has result, it means CheckSatisfiability finished.
        # However, we passed logical_not(violation). 
        # If result is None -> UNSAT (No violation found) -> Verified.
        # If result is Box -> SAT (Violation found).
        res = queue.get()
        return res
    return None

# --------------------------------------------------------------------------
# 3. Zubov Network
# --------------------------------------------------------------------------

class ZubovNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden_dim = max(32, input_dim * 3)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# --------------------------------------------------------------------------
# 4. Stress Test Loop (PINN + CEGIS)
# --------------------------------------------------------------------------

def train_and_verify_zubov(dim):
    print(f"\n{'='*40}")
    print(f"ZUBOV STRESS TEST: Dimension {dim}")
    print(f"{'='*40}")

    torch.manual_seed(MASTER_SEED + dim)
    random.seed(MASTER_SEED + dim)

    global CURRENT_MATRIX
    CURRENT_MATRIX = get_dense_system_matrix(dim)
    
    # Config
    BATCH_SIZE = 2000
    EPOCHS = 3000
    CHECK_INTERVAL = 200
    
    model = ZubovNetwork(dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # dReal vars
    dreal_vars = [Variable(f"x_{i}") for i in range(dim)]
    config = Config()
    config.use_polytope = True
    config.precision = 1e-3
    
    start_time = time.time()
    
    # Initial Data
    x_train = torch.rand((BATCH_SIZE, dim)) * 4.0 - 2.0 # [-2, 2]
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        x = x_train.requires_grad_(True)
        V = model(x)
        
        # PINN Loss: Minimize residual of Zubov PDE
        # Lie_V + phi(x)(1 - V) = 0
        
        dVdx = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
        f_x = f_value(x)
        lie_V = (f_x * dVdx).sum(dim=1, keepdim=True)
        
        phi = 0.1 * (x**2).sum(dim=1, keepdim=True)
        zubov_target = -phi * (1.0 - V)
        
        # Boundary condition: V(0) = 0
        V_0 = model(torch.zeros(1, dim))
        
        loss = ((lie_V - zubov_target)**2).mean() + 10.0*(V_0**2).mean()
        
        loss.backward()
        optimizer.step()
        
        if epoch % CHECK_INTERVAL == 0 and epoch > 0:
            print(f"  Epoch {epoch} | Loss: {loss.item():.6f}")
            
            # CEGIS Verification
            # Check if V satisfies condition on a shell [0.1, 1.5]
            # (We check a smaller region than training to ensure stability first)
            
            # Build symbolic V
            inputs = [Expression(v) for v in dreal_vars]
            V_sym_list = build_dreal_layer(model.net, inputs)
            V_sym = V_sym_list[0]
            f_sym = f_dreal(dreal_vars)
            
            print(f"  > Verifying Zubov Condition...")
            verify_start = time.time()
            
            # Check for violations using TIMEOUT_SEC
            result = CheckZubovCondition(dreal_vars, f_sym, V_sym, 0.1, 1.5, config, timeout=TIMEOUT_SEC)
            verify_time = time.time() - verify_start
            
            if result == "TIMEOUT":
                print(f"  !!! TIMEOUT ({verify_time:.2f}s) !!!")
                return "TIMEOUT"
            elif result:
                # Violation found (it's a Box)
                print(f"  [UNSAT] Violation found ({verify_time:.2f}s) -> Adding Data")
                
                # Add Counter-Example points
                # Parse Box similar to previous logic
                keys = sorted([k for k in result.keys()], key=lambda v: str(v))
                if len(keys) == dim:
                    new_pts = []
                    for _ in range(50):
                        pt = []
                        for k in keys:
                            pt.append(random.uniform(result[k].lb(), result[k].ub()))
                        new_pts.append(pt)
                    x_new = torch.tensor(new_pts, dtype=torch.float32)
                    x_train = torch.cat([x_train, x_new], dim=0)
            else:
                # No violation found (result is None)
                print(f"  [SAT] Verified! ({verify_time:.2f}s)")
                return "VERIFIED"

    return "NOT_CONVERGED"

if __name__ == "__main__":
    dimensions = [2, 4, 6, 8, 10, 20, 50, 100]
    
    print(f"\n{'='*40}")
    print(f"ZUBOV STRESS TEST (Neural Network + dReal)")
    print(f"Timeout: {TIMEOUT_SEC}s | Seed: {MASTER_SEED}")
    print(f"{'='*40}")
    
    results = {}
    
    for d in dimensions:
        res = train_and_verify_zubov(d)
        results[d] = res
        if res == "TIMEOUT":
            print(f"Stopping early at Dim {d}")
            break
            
    print("\n--- Final Results (Zubov PINN) ---")
    for d, res in results.items():
        print(f"Dim {d}: {res}")