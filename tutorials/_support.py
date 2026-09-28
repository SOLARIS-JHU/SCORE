"""Small shared numerical utilities; teaching algorithms live in the notebooks."""
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.linalg import solve_continuous_lyapunov

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.van_der_pol.EVT.commons import VanDerPol, ODEGramMatrixLyapunov


def setup(seed=42):
    """Use small, repeatable CPU workloads."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    return np.random.default_rng(seed)


def quadratic_matrix(mu=1.0):
    """Solve A.T P + P A = -I for the reverse-time linearization."""
    if mu <= 0:
        raise ValueError("mu must be positive for this stable linearization")
    A = np.array([[0.0, -1.0], [1.0, -mu]])
    return torch.tensor(solve_continuous_lyapunov(A.T, -np.eye(2)), dtype=torch.float32)


def warm_start():
    """Embed the local quadratic candidate in the repository's ten-feature model."""
    model = ODEGramMatrixLyapunov()
    with torch.no_grad():
        model.L_factor.zero_()
        model.L_factor[:2, :2].copy_(torch.linalg.cholesky(quadratic_matrix()))
    return model


def lie_derivative(model, points, physics=None, create_graph=False):
    """Return V and grad(V) dot f; preserve a graph when training/searching."""
    if physics is None:
        physics = VanDerPol()
    x = points.detach().clone().requires_grad_(True)
    value = model(x).reshape(-1)
    grad = torch.autograd.grad(value.sum(), x, create_graph=create_graph)[0]
    derivative = (grad * physics.vector_field(x)).sum(dim=1)
    return value, derivative, x


def grid(limit=2.2, size=100):
    axis = np.linspace(-limit, limit, size)
    X, Y = np.meshgrid(axis, axis)
    points = torch.tensor(np.column_stack((X.ravel(), Y.ravel())), dtype=torch.float32)
    return X, Y, points


def sample_shell(model, count, rho_core, rho, seed=43, batch_size=4096,
                 max_batches=200):
    """Uniform rejection samples from the complete shell, with a finite budget.

    Since the first two features are x and y, V >= lambda_min(Q) ||x||^2.
    Its bound supplies a containing box; no unreported spatial clipping occurs.
    Rejection becomes inefficient for ill-conditioned candidates or large rho.
    """
    if count < 1 or not 0 < rho_core < rho:
        raise ValueError("require count >= 1 and 0 < rho_core < rho")
    smallest = torch.linalg.eigvalsh(model.get_Q().detach().double()).min().item()
    if not np.isfinite(smallest) or smallest <= 0:
        raise ValueError("candidate Gram matrix must be positive definite")
    # A sharper bound follows from Q >= diag(b, b, 0, ..., 0).
    # Its maximal b is the smallest eigenvalue of the Schur complement.
    Q = model.get_Q().detach().double()
    schur = Q[:2, :2] - Q[:2, 2:] @ torch.linalg.solve(Q[2:, 2:], Q[2:, :2])
    bound = torch.linalg.eigvalsh(schur).min().item()
    radius = np.sqrt(rho / (0.99 * bound))  # small numerical safety margin
    generator = torch.Generator().manual_seed(seed)
    chunks, accepted = [], 0
    with torch.no_grad():
        for _ in range(max_batches):
            points = (2 * torch.rand(batch_size, 2, generator=generator) - 1) * radius
            values = model(points).reshape(-1)
            chosen = points[(values >= rho_core) & (values <= rho)]
            chunks.append(chosen)
            accepted += len(chosen)
            if accepted >= count:
                return torch.cat(chunks)[:count]
    raise RuntimeError(f"Only {accepted}/{count} points accepted; increase max_batches")
