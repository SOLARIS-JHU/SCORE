import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dreal import *
import time

from functions import CheckLyapunov, AddCounterexamples, build_dreal_icnn

# --------------------------------------------------------------------------
# 1. System Dynamics (Van der Pol - Reverse time)
# --------------------------------------------------------------------------

def f_value(x):
    """PyTorch dynamics for training: f(x)"""
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    
    dx1 = -x2
    dx2 = x1 + (x1**2 - 1)*x2
    return torch.cat([dx1, dx2], dim=1)

def f_dreal(vars_):
    """dReal dynamics for verification"""
    x1 = vars_[0]
    x2 = vars_[1]
    
    dx1 = -x2
    dx2 = x1 + (x1**2 - 1)*x2
    return [dx1, dx2]

# --------------------------------------------------------------------------
# 2. ICNN Model Architecture
# --------------------------------------------------------------------------

class CertifiableICNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        h = 16 
        
        # W matrices: Multiply 'x' (skip connections)
        self.W = nn.ParameterList([
            nn.Parameter(torch.Tensor(h, input_dim)), # Layer 0
            nn.Parameter(torch.Tensor(h, input_dim)), # Layer 1
            nn.Parameter(torch.Tensor(1, input_dim))  # Output
        ])
        
        # U matrices: Multiply hidden states (Must be non-negative)
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
        
        # Layer 1: z = act( W1*x + b1 + softplus(U0)*z )
        z = self.act(F.linear(x, self.W[1], self.bias[1]) + F.linear(z, F.softplus(self.U[0])))
        
        # Output: out = W2*x + b2 + softplus(U1)*z
        return F.linear(x, self.W[2], self.bias[2]) + F.linear(z, F.softplus(self.U[1]))

    def forward(self, x):
        y = self._icnn_pass(x)
        # Ensure V(0) = 0
        y_zero = self._icnn_pass(torch.zeros_like(x))
        
        # Enforce Quadratic Lower Bound for Positive Definiteness
        V_quad = 0.1 * (x**2).sum(dim=1, keepdim=True)
        
        return (y - y_zero) + V_quad

# --------------------------------------------------------------------------
# 3. Training & Verification Loop
# --------------------------------------------------------------------------

def main(plot=False):
    # Configuration
    BATCH_SIZE = 500
    EPOCHS = 2000
    CHECK_INTERVAL = 100
    BALL_LB = 0.1
    BALL_UB = 2.0 
    EPSILON = 0.01 
    LEARNING_RATE = 0.005 
    
    dim = 2

    # dReal Configuration
    config = Config()
    config.use_polytope = True
    config.precision = 1e-3 # Relaxed precision for faster ICNN checks
    x1_sym = Variable("x1")
    x2_sym = Variable("x2")
    dreal_vars = [x1_sym, x2_sym]

    # Initialize Model
    model = CertifiableICNN(input_dim=dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Initial Training Data
    x_train = torch.FloatTensor(BATCH_SIZE, dim).uniform_(-BALL_UB, BALL_UB)
    
    print(f"Starting training ICNN on Van der Pol. Domain: [{BALL_LB}, {BALL_UB}]")

    start_time = time.time()
    verified = False

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # 1. Forward Pass
        x = x_train.requires_grad_(True)
        V = model(x)
        
        # 2. Compute Derivatives (Lie Derivative)
        dVdx = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
        f_x = f_value(x)
        Lie_V = (f_x * dVdx).sum(dim=1, keepdim=True)
        
        # 3. Loss Calculation
        # ICNN guarantees V is convex, but we still need to encourage V > 0 explicitly 
        # via the V_quad term in forward().
        # Main objective: Lie_V <= -EPSILON
        
        loss_lie = torch.relu(Lie_V + EPSILON).mean()
        loss_pos = torch.relu(-V).mean()
        
        zero_input = torch.zeros(1, dim, requires_grad=True)
        V_zero = model(zero_input)
        grad_zero = torch.autograd.grad(V_zero, zero_input, create_graph=True)[0]
        loss_origin_grad = (grad_zero**2).sum()
        
        # Small regularization on V to keep it smooth
        loss = loss_lie + 1e-4 * V.mean() + 10.0*loss_pos + 1.0 * loss_origin_grad
        
        loss.backward()
        optimizer.step()
        
        if epoch % CHECK_INTERVAL == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f} | Data Size: {len(x_train)}")
            
            # ------------------------------------------------------
            # Verification Step (dReal)
            # ------------------------------------------------------
            
            # Reconstruct symbolic V and its gradient from current ICNN weights
            nn_sym, nn_grad = build_dreal_icnn(model, dreal_vars)
            
            # Add the quadratic lower bound part symbolically
            # V(x) = NN(x) + 0.1*||x||^2
            # grad(x) = NN_grad(x) + 0.2*x
            quad_sym = 0.1 * (x1_sym*x1_sym + x2_sym*x2_sym)
            
            V_sym = nn_sym + quad_sym
            grad_sym = [nn_grad[i] + 0.2*dreal_vars[i] for i in range(dim)]
            
            f_sym = f_dreal(dreal_vars)
            
            print("  Verifying ICNN...")
            # We pass grad_sym to CheckLyapunov for efficiency
            result = CheckLyapunov(dreal_vars, f_sym, V_sym, grad_sym, BALL_LB, BALL_UB, config, EPSILON)
            
            if result:
                print("  [UNSAT] Counter-example found. Adding to training set.")
                x_train = AddCounterexamples(x_train, result, N=50)
            else:
                print("  [SAT] Verified! No counter-examples found.")
                verified = True
                break

    end_time = time.time()
    print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")
    print(f"Verified: {verified}")

    # --------------------------------------------------------------------------
    # 4. Saving Results & Plotting
    # --------------------------------------------------------------------------
    
    if verified or plot:
        print("Generating plots...")
        x_grid = np.linspace(-2.5, 2.5, 200)
        y_grid = np.linspace(-2.5, 2.5, 200)
        X, Y = np.meshgrid(x_grid, y_grid)
        inputs = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
        
        with torch.no_grad():
            V_plot = model(inputs).numpy().reshape(X.shape)

        inputs.requires_grad = True
        V_tens = model(inputs)
        grads = torch.autograd.grad(V_tens.sum(), inputs, create_graph=False)[0]
        f_tens = f_value(inputs)
        Lie_plot = (f_tens * grads).sum(dim=1).detach().numpy().reshape(X.shape)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        c1 = ax[0].contourf(X, Y, V_plot, levels=20, cmap='viridis')
        fig.colorbar(c1, ax=ax[0])
        ax[0].set_title("ICNN Lyapunov Function V(x)")
        
        c2 = ax[1].contourf(X, Y, Lie_plot, levels=20, cmap='coolwarm')
        fig.colorbar(c2, ax=ax[1])
        ax[1].set_title("Lie Derivative")
        ax[1].contour(X, Y, Lie_plot, levels=[0], colors='red', linewidths=2)

        plt.savefig("icnn_vdp_plot.png")
        print("Plots saved to icnn_vdp_plot.png")
        
        param_filename = "vdp_icnn_params.pth"
        torch.save(model.state_dict(), param_filename)
        print(f"Model parameters saved to {param_filename}")

if __name__ == "__main__":
    main(plot=True)