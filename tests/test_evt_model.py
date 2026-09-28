"""Regression checks for the shared model and tutorial numerical identities."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tutorials'))
from _support import quadratic_matrix, warm_start, lie_derivative, sample_shell
from experiments.van_der_pol.EVT.commons import VanDerPol, ODEGramMatrixLyapunov, construct_ode_monomials


class ModelTests(unittest.TestCase):
    def test_default_model_and_origin(self):
        model = ODEGramMatrixLyapunov()
        points = torch.tensor([[0., 0.], [0.2, -0.4], [1., 1.]])
        values = model(points).flatten()
        self.assertEqual(values[0].item(), 0.)
        self.assertTrue(torch.all(values[1:] > 0))
        self.assertTrue(torch.all(values >= 1e-4 * points.square().sum(1) - 1e-7))
        with self.assertRaises(ValueError):
            ODEGramMatrixLyapunov(feature_dim=5)

    def test_mu_controls_reverse_dynamics(self):
        x = torch.tensor([[0.5, 0.4]])
        self.assertTrue(torch.allclose(VanDerPol(mu=2).vector_field(x),
                                       torch.tensor([[-0.4, -0.1]]), atol=1e-7))

    def test_large_features_and_gradients_are_finite(self):
        x = torch.tensor([[100., -100.]], requires_grad=True)
        z = construct_ode_monomials(x)
        self.assertTrue(torch.isfinite(z).all())
        self.assertTrue(torch.isfinite(torch.autograd.grad(z.sum(), x)[0]).all())
        small = torch.tensor([[0.3, -0.4]])
        self.assertTrue(torch.allclose(construct_ode_monomials(small)[:, 5:7],
                                       torch.log(torch.cosh(small)), atol=1e-7))

    def test_linear_lyapunov_identity_and_nonlinear_derivative(self):
        P = quadratic_matrix()
        A = torch.tensor([[0., -1.], [1., -1.]])
        self.assertTrue(torch.allclose(A.T @ P + P @ A, -torch.eye(2), atol=1e-6))
        points = torch.tensor([[0.2, 0.4], [-0.8, 0.3]])
        candidate = lambda x: torch.einsum('bi,ij,bj->b', x, P, x)
        _, derivative, _ = lie_derivative(candidate, points)
        exact = (2 * (points @ P) * VanDerPol().vector_field(points)).sum(1)
        self.assertTrue(torch.allclose(derivative, exact, atol=1e-6))

    def test_shell_sampling_and_budget(self):
        model = warm_start()
        points = sample_shell(model, 200, 0.03, 0.6)
        values = model(points).flatten()
        self.assertTrue(((values >= 0.03) & (values <= 0.6)).all())
        self.assertTrue(torch.equal(points, sample_shell(model, 200, 0.03, 0.6)))
        with self.assertRaises(ValueError):
            sample_shell(model, 10, 0.6, 0.03)
        with self.assertRaises(RuntimeError):
            sample_shell(model, 10, 0.03, 0.6, max_batches=0)

    def test_existing_checkpoint_compatibility(self):
        root = Path(__file__).resolve().parents[1]
        checkpoint = torch.load(root / 'experiments/van_der_pol/EVT/models/lyapunov_model.pth',
                                map_location='cpu', weights_only=True)
        model = ODEGramMatrixLyapunov()
        model.load_state_dict(checkpoint)
        self.assertTrue(torch.isfinite(model(torch.tensor([[0.3, -0.2]]))).all())


if __name__ == '__main__':
    unittest.main()
