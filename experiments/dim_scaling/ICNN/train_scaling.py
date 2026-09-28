import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import random
import signal
from dreal import *
from functions import CheckLyapunov, build_dreal_icnn

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
MASTER_SEED = 42
TIMEOUT_SEC = 600  # 10 minutes limit per verification
TRAIN_EPOCHS = 2000

# --------------------------------------------------------------------------
# 1. Dynamics & Model
# --------------------------------------------------------------------------
def get_dense_system_matrix(dim):
    A = np.zeros((dim, dim))
    for i in range(0, dim, 2):
        A[i, i+1] = 1.0
        A[i+1, i] = -1.0
        A[i+1, i+1] = -0.5
    np.random.seed(MASTER_SEED + dim)
    H = np.random.randn(dim, dim)
    Q, _ = np.linalg.qr(H)
    return torch.tensor(Q @ A @ Q.T, dtype=torch.float32)

CURRENT_MATRIX = None 
def f_value(x): return x @ CURRENT_MATRIX.t()
def f_dreal(vars_):
    M_np = CURRENT_MATRIX.numpy()
    return [sum(float(M_np[i,j]) * vars_[j] for j in range(len(vars_))) for i in range(len(vars_))]

class CertifiableICNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        h = 16 
        
        # All W matrices multiply 'x', so they must all accept 'input_dim'
        self.W = nn.ParameterList([
            nn.Parameter(torch.Tensor(h, input_dim)), # Layer 0: x -> h1
            nn.Parameter(torch.Tensor(h, input_dim)), # Layer 1: x -> h2
            nn.Parameter(torch.Tensor(1, input_dim))  # Output:  x -> out
        ])
        
        # U matrices map hidden -> hidden, so they use 'h'
        self.U = nn.ParameterList([
            nn.Parameter(torch.Tensor(h, h)), # h1 -> h2
            nn.Parameter(torch.Tensor(1, h))  # h2 -> out
        ])
        
        self.bias = nn.ParameterList([
            nn.Parameter(torch.Tensor(h)),
            nn.Parameter(torch.Tensor(h)),
            nn.Parameter(torch.Tensor(1))
        ])
        
        self.act = nn.Softplus()
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.uniform_(p, -0.1, 0.1)

    def _icnn_pass(self, x):
        # Layer 0
        z = self.act(F.linear(x, self.W[0], self.bias[0]))
        
        # Layer 1: U maps z->z, W maps x->z
        z = self.act(F.linear(x, self.W[1], self.bias[1]) + F.linear(z, F.softplus(self.U[0])))
        
        # Output: U maps z->out, W maps x->out
        return F.linear(x, self.W[2], self.bias[2]) + F.linear(z, F.softplus(self.U[1]))

    def forward(self, x):
        y = self._icnn_pass(x)
        y_zero = self._icnn_pass(torch.zeros_like(x))
        V_quad = 0.01 * (x**2).sum(dim=1, keepdim=True)
        return (y - y_zero) + V_quad

# --------------------------------------------------------------------------
# 2. Timeout Handler
# --------------------------------------------------------------------------
class TimeoutException(Exception): pass
def handler(signum, frame): raise TimeoutException()

# --------------------------------------------------------------------------
# 3. Main Routine: Train -> Verify
# --------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_certify(dim):
    print(f"\n[{dim}D] Training Neural Lyapunov ({TRAIN_EPOCHS} epochs) on {device}...")
    
    torch.manual_seed(MASTER_SEED + dim)
    random.seed(MASTER_SEED + dim)
    
    # ---------------------------------------------------------
    # 1. SETUP & TRANSFER TO GPU
    # ---------------------------------------------------------
    global CURRENT_MATRIX
    CURRENT_MATRIX = get_dense_system_matrix(dim).to(device)
    
    model = CertifiableICNN(dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    x_train = torch.FloatTensor(3000, dim).uniform_(-1.0, 1.0).to(device)
    
    # ---------------------------------------------------------
    # 2. TRAINING
    # ---------------------------------------------------------
    for epoch in range(TRAIN_EPOCHS):
        optimizer.zero_grad()
        
        x = x_train.requires_grad_(True)
        
        V = model(x)
        dVdx = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
        
        Lie_V = (f_value(x) * dVdx).sum(1, keepdim=True)
        
        loss = torch.relu(Lie_V + 0.05).mean() + 0.1*torch.relu(-V).mean()
        loss.backward()
        optimizer.step()
    
    print(f"[{dim}D] Training Done. Loss: {loss.item():.5f}")
    
    # ---------------------------------------------------------
    # 3. TRANSFER BACK TO CPU (CRITICAL FOR DREAL)
    # ---------------------------------------------------------
    # dReal/SMT cannot handle CUDA tensors. We must move everything back.
    model.cpu()
    CURRENT_MATRIX = CURRENT_MATRIX.cpu()
    
    # ---------------------------------------------------------
    # 4. CERTIFY (ON CPU)
    # ---------------------------------------------------------
    print(f"[{dim}D] Starting SMT Verification...")
    
    dreal_vars = [Variable(f"x_{i}") for i in range(dim)]
    config = Config()
    config.use_polytope = True
    config.precision = 0.01 
    
    # Build Symbolic Math
    nn_sym, nn_grad = build_dreal_icnn(model, dreal_vars)
    quad_sym = 0.01 * sum(v*v for v in dreal_vars)
    
    # Run a pass on CPU to get the zero value
    y_zero = model._icnn_pass(torch.zeros(1, dim)).item()
    
    V_sym = (nn_sym - y_zero) + quad_sym
    grad_sym = [nn_grad[i] + 0.02*dreal_vars[i] for i in range(dim)]
    
    # f_dreal uses CURRENT_MATRIX.numpy(), which requires CPU
    f_sym = f_dreal(dreal_vars)

    start_time = time.time()
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(TIMEOUT_SEC)
    
    try:
        res = CheckLyapunov(dreal_vars, f_sym, V_sym, grad_sym, 0.1, 1.0, config, 0.001, timeout=TIMEOUT_SEC)
        elapsed = time.time() - start_time
        signal.alarm(0) 
        
        if res:
            return f"FAILED (Counter-Example found in {elapsed:.2f}s)"
        else:
            return f"SUCCESS (Verified in {elapsed:.2f}s)"
            
    except TimeoutException:
        return f"TIMEOUT (Exceeded {TIMEOUT_SEC}s)"
    except Exception as e:
        return f"CRASH ({str(e)})"

if __name__ == "__main__":
    dims = [2, 4, 6, 8, 10, 20, 50, 100]
    
    print(f"Benchmarking SMT Scalability (Timeout: {TIMEOUT_SEC}s)")
    print("Goal: Show that Low Dim = Success, High Dim = Timeout")
    
    for d in dims:
        result = train_and_certify(d)
        print(f"Result for Dim {d}: {result}")
        
        # Break if we hit timeout to save time
        if "TIMEOUT" in result:
            print("--> Timeout reached. Stopping benchmark.")
            break