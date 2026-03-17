import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.linalg
import os
import random
from commons import VanDerPol, ODEGramMatrixLyapunov


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f" [INFO] Seed set to {seed}")

# ==========================================
# 1. HELPERS FOR LINEARIZATION 
# ==========================================
def solve_lyapunov_equation(mu):
    A = np.array([[0, -1],
                  [1, -mu]])
    Q_cost = np.eye(2)
    P = scipy.linalg.solve_continuous_lyapunov(A.T, -Q_cost)
    return torch.tensor(P, dtype=torch.float32)

def initialize_with_lqr(model, physics):
    print(" [INIT] Solving Linear Lyapunov Equation for warm-start...")
    P = solve_lyapunov_equation(physics.mu)
    L_ideal = torch.linalg.cholesky(P)
    with torch.no_grad():
        model.L_factor.fill_(0.0)
        model.L_factor[0:2, 0:2] = L_ideal.to(model.device)
        model.L_factor[2:, 2:] = torch.randn_like(model.L_factor[2:, 2:]) * 0.01

# ==========================================
# 2. TRAINING LOOP
# ==========================================
def train_lyapunov(n_epochs=2000, batch_size=200, lr=0.01, time_reverse=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    physics = VanDerPol(mu=1.0, device=device)
    model = ODEGramMatrixLyapunov(state_dim=2, feature_dim=10, device=device).to(device)
    
    if time_reverse:
        initialize_with_lqr(model, physics)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)
    
    x_range = [-2, 2] 
    y_range = [-3, 3]
    loss_history = []

    print(f"--- Starting Training ---")

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        x = (torch.rand(batch_size, 1, device=device) * (x_range[1]-x_range[0])) + x_range[0]
        y = (torch.rand(batch_size, 1, device=device) * (y_range[1]-y_range[0])) + y_range[0]
        u = torch.cat([x, y], dim=1)
        u.requires_grad_(True)

        V = model(u)
        grads = torch.autograd.grad(V.sum(), u, create_graph=True)[0]
        
        f_u = physics.vector_field(u)
        if time_reverse: f_u = -f_u
        V_dot = torch.sum(grads * f_u, dim=1, keepdim=True)

        # Loss
        margin = 0.1
        violation = torch.relu(V_dot + margin * V)
        target_shape = torch.sum(u**2, dim=1, keepdim=True)
        shape_loss = torch.mean((V - target_shape)**2)
        loss = torch.mean(violation) + 0.5 * shape_loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    return model, loss_history

# ==========================================
# 3. UPDATED VISUALIZATION
# ==========================================
def visualize_landscape(model, physics, time_reverse=False):
    model.eval()
    
    # 1. Generate Grid
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    pts_tensor = torch.tensor(pts, dtype=torch.float32, device=model.device)
    
    pts_tensor.requires_grad_(True)
    V = model(pts_tensor)
    
    # 2. Compute V_dot
    grads = torch.autograd.grad(outputs=V.sum(), inputs=pts_tensor)[0]
    f_u = physics.vector_field(pts_tensor)
    if time_reverse: f_u = -f_u
    V_dot = torch.sum(grads * f_u, dim=1)

    # 3. Convert to Numpy
    V_np = V.detach().cpu().numpy().reshape(X.shape)
    V_dot_np = V_dot.detach().cpu().numpy().reshape(X.shape)

    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    # We want small levels (0 to 2.5) to see the inner details
    custom_levels = np.linspace(0.1, 2.5, 12) 

    # ==========================================
    # PLOT 1: Lyapunov Function V(x)
    # ==========================================
    # Background V(x)
    c1 = ax[0].contourf(X, Y, V_np, levels=50, cmap='viridis')
    plt.colorbar(c1, ax=ax[0])

    # Level Sets (White lines)
    # Using 'custom_levels' to force focus on values < 2.5
    levels_lines = ax[0].contour(X, Y, V_np, levels=custom_levels, colors='white', linewidths=0.8)
    ax[0].clabel(levels_lines, inline=True, fontsize=8, fmt='%.1f')

    # Streamlines
    skip = 10
    flow_u = -f_u if time_reverse else f_u
    flow_np = flow_u.detach().cpu().numpy()
    ax[0].streamplot(X[::skip, ::skip], Y[::skip, ::skip], 
                     flow_np[:,0].reshape(X.shape)[::skip, ::skip], 
                     flow_np[:,1].reshape(X.shape)[::skip, ::skip], 
                     color=(1, 1, 1, 0.3), linewidth=0.5, arrowsize=1.0)
    
    ax[0].set_title("Learned Lyapunov Function V(x)\n(White lines = V(x) Level Sets)", fontsize=12)

    # ==========================================
    # PLOT 2: Lie Derivative V_dot(x)
    # ==========================================
    # Background V_dot(x)
    vmin, vmax = V_dot_np.min(), V_dot_np.max()
    if vmin >= 0: vmin = -0.1
    if vmax <= 0: vmax = 0.1
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
    c2 = ax[1].contourf(X, Y, V_dot_np, levels=50, cmap='RdYlGn_r', norm=norm)
    plt.colorbar(c2, ax=ax[1])

    # V_dot Zero Level Set (Stability Boundary) - Black Dashed
    zero_contour = ax[1].contour(X, Y, V_dot_np, levels=[0], colors='black', linewidths=2.5, linestyles='--')
    ax[1].clabel(zero_contour, inline=True, fontsize=10, fmt='V_dot=0')

    # We plot the EXACT SAME V(x) levels (custom_levels) on the V_dot plot.
    # We use 'cyan' to make them stand out against the Red/Green background.
    v_overlay = ax[1].contour(X, Y, V_np, levels=custom_levels, colors='cyan', linewidths=1.0, linestyles='-')
    ax[1].clabel(v_overlay, inline=True, fontsize=8, fmt='V=%.1f')

    ax[1].set_title("Lie Derivative V_dot(x)\n(Cyan lines = V(x) Level Sets Overlay)", fontsize=12)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    # plt.savefig('plots/training_result_overlay.png')
    # plt.show()

if __name__ == "__main__":
    set_seed(42)
    # 1. Train
    trained_model, _ = train_lyapunov()
    
    # 2. Save Model
    # os.makedirs('models', exist_ok=True)

    # Save the state dictionary (weights)
    torch.save(trained_model.state_dict(), 'models/lyapunov_model.pth')
    print("Model saved to models/lyapunov_model.pth")

    # 3. Visualize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pde = VanDerPol(device=device)
    visualize_landscape(trained_model, pde)