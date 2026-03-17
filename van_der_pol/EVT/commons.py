import torch
import torch.nn as nn
import numpy as np

# ==========================================
# 1. ODE Dynamics: Van der Pol Oscillator
#    dx/dt = y
#    dy/dt = \mu(1 - x^2)y - x
# ==========================================

class VanDerPol:
    def __init__(self, mu=1.0, device='cpu'):
        self.mu = mu
        self.device = device
    
    def vector_field(self, u):
        """
        Matches 'f_value' from Neural Lyapunov Notebook:
        dx1 = -x2
        dx2 = x1 + (x1^2 - 1)*x2
        """
        x = u[:, 0:1]
        y = u[:, 1:2]
        
        dx = -y
        dy = x + (x**2 - 1) * y 
        
        return torch.cat([dx, dy], dim=1)

    def solve(self, u0, T=15.0, dt=0.01):
        """
        Simple RK4 solver for visualization.
        u0: torch tensor (1, 2)
        """
        u = u0.clone()
        history = [u.detach().cpu().numpy().flatten()]
        times = [0.0]
        num_steps = int(T / dt)
        
        for i in range(num_steps):
            k1 = self.vector_field(u)
            k2 = self.vector_field(u + 0.5 * dt * k1)
            k3 = self.vector_field(u + 0.5 * dt * k2)
            k4 = self.vector_field(u + dt * k3)
            
            u = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            if i % 10 == 0:
                history.append(u.detach().cpu().numpy().flatten())
                times.append((i+1)*dt)
        
        return np.array(times), np.array(history)


# ==========================================
# 2. SINDy-Style Candidate
# ==========================================

def construct_ode_monomials(u):
    x = u[:, 0:1]
    y = u[:, 1:2]
    
    # --- 1. Polynomial Backbone---
    p_poly = [x, y, x**2, x*y, y**2]
    
    # --- 2. "Leaky" Non-Polynomials ---
    lc_x = torch.log(torch.cosh(x))
    lc_y = torch.log(torch.cosh(y))
    
    # B. Modulated Tanh (Interaction Terms)
    x_thy = x * torch.tanh(y)  # Acts like xy near 0, x*sgn(y) far away
    y_thx = y * torch.tanh(x)  # Acts like yx near 0, y*sgn(x) far away
    
    # C. Mixed "Stiff" Term
    stiff_term = x * torch.tanh(x) * y 

    # Concatenate all
    return torch.cat(p_poly + [lc_x, lc_y, x_thy, y_thx, stiff_term], dim=1)


class ODEGramMatrixLyapunov(nn.Module):
    def __init__(self, state_dim=2, feature_dim=5, device='cpu'):
        super().__init__()
        self.device = device
        
        # Initialize L close to Identity to ensure V starts convex (circular)
        # This helps training significantly vs random initialization
        self.L_factor = nn.Parameter(torch.randn(feature_dim, feature_dim, device=device) * 0.01)
        
        # Bias the diagonal elements to be positive initially
        with torch.no_grad():
            self.L_factor.add_(torch.eye(feature_dim, device=device) * 0.1)

        self.register_buffer('eye', torch.eye(feature_dim) * 1e-4)

    def get_Q(self):
        L = self.L_factor
        return L @ L.T + self.eye

    def forward(self, u):
        z = construct_ode_monomials(u)
        Q = self.get_Q()
        density = torch.einsum('bf, fg, bg -> b', z, Q, z)
        return density.view(-1, 1)