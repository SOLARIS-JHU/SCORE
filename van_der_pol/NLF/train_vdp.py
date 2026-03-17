import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dreal import *
import time
import os

# Import our helper functions
from functions import CheckLyapunov, AddCounterexamples, build_dreal_lyapunov

# --------------------------------------------------------------------------
# 1. System Dynamics (Van der Pol - Reverse time for stability at origin)
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
# 2. Neural Lyapunov Model
# --------------------------------------------------------------------------

class LyapunovFunction(nn.Module):
    def __init__(self):
        super(LyapunovFunction, self).__init__()
        # Architecture: 2 inputs -> 10 hidden -> 10 hidden -> 1 output
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 1, bias=False) 
        self.act = nn.Tanh()

    def forward(self, x):
        h1 = self.act(self.layer1(x))
        h2 = self.act(self.layer2(h1))
        out = self.layer3(h2)
        
        # Enforce Positive Definiteness: V(x) = NN(x)^2 + 0.1*||x||^2
        # This ensures V(0)=0 and V(x) > 0 for x != 0
        V_quad = 0.1 * (x**2).sum(dim=1, keepdim=True)
        return out*out + V_quad

# --------------------------------------------------------------------------
# 3. Training & Verification Loop
# --------------------------------------------------------------------------

def main(plot=False):
    # Configuration
    BATCH_SIZE = 500
    EPOCHS = 2000
    CHECK_INTERVAL = 100
    BALL_LB = 0.1
    BALL_UB = 2.0  # Region of attraction radius
    EPSILON = 0.01 # Tolerance for Lie derivative (Lie_V < -EPSILON)
    LEARNING_RATE = 0.01

    # dReal Configuration
    config = Config()
    config.use_polytope = True
    config.precision = 1e-5
    x1_sym = Variable("x1")
    x2_sym = Variable("x2")
    dreal_vars = [x1_sym, x2_sym]

    # Initialize Model
    model = LyapunovFunction()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Initial Training Data (Uniform sampling)
    x_train = torch.FloatTensor(BATCH_SIZE, 2).uniform_(-BALL_UB, BALL_UB)
    
    print(f"Starting training on Van der Pol. Domain: [{BALL_LB}, {BALL_UB}]")

    start_time = time.time()
    verified = False

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # 1. Forward Pass
        x = x_train.requires_grad_(True)
        V = model(x)
        
        # 2. Compute Derivatives (Lie Derivative)
        # dV/dx
        dVdx = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
        # f(x)
        f_x = f_value(x)
        # Lie_V = dV/dx * f(x)
        Lie_V = (f_x * dVdx).sum(dim=1, keepdim=True)
        
        # 3. Loss Calculation
        # Condition 1: V(x) >= 0 (Implicitly satisfied by architecture, but good to monitor)
        # Condition 2: Lie_V(x) <= -EPSILON
        
        # We penalize points where Lie_V > -EPSILON
        loss_lie = torch.relu(Lie_V + EPSILON).mean()
        
        loss = loss_lie
        
        loss.backward()
        optimizer.step()
        
        if epoch % CHECK_INTERVAL == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f} | Data Size: {len(x_train)}")
            
            # ------------------------------------------------------
            # Verification Step (dReal)
            # ------------------------------------------------------
            
            # Reconstruct symbolic V from current model weights
            # V_sym = (NN_sym)^2 + 0.1*(x1^2 + x2^2)
            nn_sym = build_dreal_lyapunov(model, dreal_vars)
            quad_sym = 0.1 * (x1_sym*x1_sym + x2_sym*x2_sym)
            V_sym = nn_sym * nn_sym + quad_sym
            
            f_sym = f_dreal(dreal_vars)
            
            # Check conditions
            print("  Verifying...")
            result = CheckLyapunov(dreal_vars, f_sym, V_sym, BALL_LB, BALL_UB, config, EPSILON)
            
            if result:
                print("  [UNSAT] Counter-example found. Adding to training set.")
                # Add counter-examples to training data to "fix" the model there
                x_train = AddCounterexamples(x_train, result, N=50)
            else:
                print("  [SAT] Verified! No counter-examples found.")
                verified = True
                break

    end_time = time.time()
    print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")
    print(f"Verified: {verified}")

    # --------------------------------------------------------------------------
    # 4. Saving Results
    # --------------------------------------------------------------------------
    
    # A. Save Model Parameters
    param_filename = "vdp_lyapunov_params.pth"
    torch.save(model.state_dict(), param_filename)
    print(f"Model parameters saved to {param_filename}")

    # B. Generate and Save Plot
    print("Generating plots...")
    
    # Create grid
    x_grid = np.linspace(-2.5, 2.5, 200)
    y_grid = np.linspace(-2.5, 2.5, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    inputs = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
    
    
    # 1. Calculate V(x) (No gradients needed here)
    with torch.no_grad():
        V_plot = model(inputs).numpy().reshape(X.shape)

    # 2. Calculate Lie Derivative
    inputs.requires_grad = True
    V_tens = model(inputs)
    grads = torch.autograd.grad(V_tens.sum(), inputs, create_graph=False)[0]
    f_tens = f_value(inputs)
    Lie_plot = (f_tens * grads).sum(dim=1).detach().numpy().reshape(X.shape)
    
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Lyapunov Function V(x)
    c1 = ax[0].contourf(X, Y, V_plot, levels=20, cmap='viridis')
    fig.colorbar(c1, ax=ax[0])
    ax[0].set_title("Lyapunov Function V(x)")
    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")
    # Draw verification bounds
    circle = plt.Circle((0, 0), BALL_UB, color='r', fill=False, linestyle='--', label='Verify Bound')
    ax[0].add_artist(circle)
    ax[0].legend()

    # Plot 2: Lie Derivative
    c2 = ax[1].contourf(X, Y, Lie_plot, levels=20, cmap='coolwarm')
    fig.colorbar(c2, ax=ax[1])
    ax[1].set_title("Lie Derivative (Target < 0)")
    ax[1].set_xlabel("x1")
    ax[1].set_ylabel("x2")
    ax[1].contour(X, Y, Lie_plot, levels=[0], colors='red', linewidths=2) # Zero level set

    if plot == True:
        plot_filename = "vdp_lyapunov_plot.png"
        plt.savefig(plot_filename)
        print(f"Plots saved to {plot_filename}")

if __name__ == "__main__":
    main()