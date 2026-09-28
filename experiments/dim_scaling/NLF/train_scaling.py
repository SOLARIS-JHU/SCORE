import torch
import torch.nn as nn
import numpy as np
import time
import scipy.linalg
import random
from dreal import *
from functions import CheckLyapunov, AddCounterexamples, build_dreal_lyapunov

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
    A is the block-diagonal damped oscillator.
    R is a random special orthogonal rotation matrix.
    """
    # 1. Base System (Sparse)
    # Block diagonal: [[0, 1], [-1, -0.5]] repeated
    A = np.zeros((dim, dim))
    for i in range(0, dim, 2):
        A[i, i+1] = 1.0
        A[i+1, i] = -1.0
        A[i+1, i+1] = -0.5
        
    # 2. Random Rotation (Dense mixing)
    # Use MASTER_SEED + dim to ensure reproducibility per dimension
    np.random.seed(MASTER_SEED + dim)
    
    # We use QR decomposition of a random matrix to get a valid rotation Q
    H = np.random.randn(dim, dim)
    Q, _ = np.linalg.qr(H)
    
    # M is dense but has the exact same eigenvalues (stability) as A
    M = Q @ A @ Q.T
    return torch.tensor(M, dtype=torch.float32)

# Global storage for the current matrix (so dReal and PyTorch use the same one)
CURRENT_MATRIX = None 

def f_value(x):
    """
    Dense Dynamics: dx = x * M^T  (since x is [Batch, Dim])
    """
    global CURRENT_MATRIX
    return x @ CURRENT_MATRIX.t()

def f_dreal(vars_):
    """
    Dense Dynamics for dReal.
    """
    global CURRENT_MATRIX
    dim = len(vars_)
    M_np = CURRENT_MATRIX.numpy()
    
    dx = []
    for i in range(dim):
        # dot product for row i
        val = 0.0
        for j in range(dim):
            val += float(M_np[i, j]) * vars_[j]
        dx.append(val)
        
    return dx

# --------------------------------------------------------------------------
# 2. Scalable Neural Lyapunov Model
# --------------------------------------------------------------------------

class LyapunovFunction(nn.Module):
    def __init__(self, input_dim):
        super(LyapunovFunction, self).__init__()
        hidden_dim = max(32, input_dim * 3)
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1, bias=False) 
        self.act = nn.Tanh()

    def forward(self, x):
        h1 = self.act(self.layer1(x))
        h2 = self.act(self.layer2(h1))
        out = self.layer3(h2)
        
        # Enforce V > 0
        V_quad = 0.01 * (x**2).sum(dim=1, keepdim=True)
        return out*out + V_quad

# --------------------------------------------------------------------------
# 3. Stress Test Loop
# --------------------------------------------------------------------------

def train_and_verify(dim):
    print(f"\n{'='*40}")
    print(f"STRESS TEST (DENSE): Dimension {dim}")
    print(f"{'='*40}")

    torch.manual_seed(MASTER_SEED + dim)
    random.seed(MASTER_SEED + dim)

    # Generate the dense matrix for this run
    global CURRENT_MATRIX
    CURRENT_MATRIX = get_dense_system_matrix(dim)
    
    # Config
    BATCH_SIZE = 2000 if dim < 6 else 5000 
    EPOCHS = 3000
    CHECK_INTERVAL = 100
    BALL_LB = 0.1
    BALL_UB = 1.0 
    EPSILON = 0.01
    
    # dReal Setup
    config = Config()
    config.use_polytope = True
    config.precision = 1e-3 
    dreal_vars = [Variable(f"x_{i}") for i in range(dim)]
    
    model = LyapunovFunction(dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    x_train = torch.FloatTensor(BATCH_SIZE, dim).uniform_(-BALL_UB, BALL_UB)
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # Forward
        x = x_train.requires_grad_(True)
        V = model(x)
        
        # Lie Derivative
        dVdx = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
        f_x = f_value(x)
        Lie_V = (f_x * dVdx).sum(dim=1, keepdim=True)
        
        loss = torch.relu(Lie_V + EPSILON).mean()
        loss.backward()
        optimizer.step()
        
        if epoch % CHECK_INTERVAL == 0 and epoch > 0:
            print(f"  Epoch {epoch} | Loss: {loss.item():.5f}")
            
            # --- CEGIS LOOP ---
            nn_sym = build_dreal_lyapunov(model, dreal_vars)
            quad_sym = 0.01 * sum(v*v for v in dreal_vars)
            V_sym = nn_sym * nn_sym + quad_sym
            f_sym = f_dreal(dreal_vars)
            
            print(f"  > Verifying {dim}-D DENSE system...")
            verify_start = time.time()
            
            # Verify with global TIMEOUT_SEC
            result = CheckLyapunov(dreal_vars, f_sym, V_sym, BALL_LB, BALL_UB, config, EPSILON, timeout=TIMEOUT_SEC)
            verify_time = time.time() - verify_start
            
            if result == "TIMEOUT":
                print(f"  !!! FAILED: SMT Solver Timed Out ({verify_time:.2f}s) !!!")
                return "TIMEOUT"
            elif result:
                print(f"  [UNSAT] Counter-example found ({verify_time:.2f}s) -> Adding Data")
                # Add the hard region to training set
                x_train = AddCounterexamples(x_train, result, N=100)
            else:
                print(f"  [SAT] Verified! ({verify_time:.2f}s)")
                return "VERIFIED"

    return "NOT_CONVERGED"

if __name__ == "__main__":
    dimensions = [2, 4, 6, 8, 10, 20, 50, 100] 
    
    print(f"\n{'='*40}")
    print(f"LYAPUNOV STRESS TEST (Neural Network + dReal)")
    print(f"Timeout: {TIMEOUT_SEC}s | Seed: {MASTER_SEED}")
    print(f"{'='*40}")

    results = {}
    
    for d in dimensions:
        res = train_and_verify(d)
        results[d] = res
        
        if res == "TIMEOUT":
            print(f"\nStopping early at Dim {d} due to Timeout.")
            break
            
    print("\n--- Final Results (Dense System) ---")
    for d, res in results.items():
        print(f"Dim {d}: {res}")