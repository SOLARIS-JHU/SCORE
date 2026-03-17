import torch
import torch.nn as nn
import numpy as np
import time
import sys

# --- FIX 1: Only import FindMaxROA. Do NOT import f_dreal. ---
from functions import FindMaxROA

# --------------------------------------------------------------------------
# 0. Device Configuration
# --------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------------------------------
# 1. Dynamics (Defined LOCALLY here)
# --------------------------------------------------------------------------
def f_value(x):
    """Reversed Van der Pol Dynamics (PyTorch)"""
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    dx1 = -x2
    dx2 = x1 + (x1**2 - 1)*x2
    return torch.cat([dx1, dx2], dim=1)

def f_dreal(vars_):
    """
    Reversed Van der Pol Dynamics (dReal symbolic).
    Input: vars_ is a list [x1, x2]
    """
    x1 = vars_[0]
    x2 = vars_[1]
    dx1 = -x2
    dx2 = x1 + (x1**2 - 1)*x2
    return [dx1, dx2]

# --------------------------------------------------------------------------
# 2. PINN Model (Zubov Network)
# --------------------------------------------------------------------------
class ZubovNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 -> 10 -> 10 -> 1
        self.net = nn.Sequential(
            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.net(x)

# --------------------------------------------------------------------------
# 3. Training Loop
# --------------------------------------------------------------------------
def train_zubov():
    BATCH_SIZE = 1000
    EPOCHS = 3000  
    LR = 1e-3
    DOMAIN_LIMIT = 3.0 
    
    model = ZubovNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    print("Training PINN to solve Zubov Equation...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        x = torch.rand((BATCH_SIZE, 2), device=device) * (2 * DOMAIN_LIMIT) - DOMAIN_LIMIT
        x.requires_grad_(True)
        
        V = model(x)
        x_zero = torch.zeros((1, 2), device=device)
        V_0 = model(x_zero)
        
        dVdx = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
        f = f_value(x)
        
        phi = 0.1 * (x**2).sum(dim=1, keepdim=True)
        lie_derivative = (dVdx * f).sum(dim=1, keepdim=True)
        target = -phi * (1.0 - V)
        
        loss = ((lie_derivative - target)**2).mean() + (V_0**2).mean() 
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    print(f"Training finished in {time.time()-start_time:.2f}s")
    
    save_path = "zubov_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model parameters saved to '{save_path}'")

    model.cpu() 
    return model

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Train
    model = train_zubov()
    
    # 2. Verification
    # We pass the LOCAL 'f_dreal' function to the external FindMaxROA helper
    print("\nStarting Verification...")
    best_c = FindMaxROA(model, f_dreal, lb=0.1, ub=3.0)
    
    print(f"\nFinal Certified ROA Level Set: c = {best_c}")