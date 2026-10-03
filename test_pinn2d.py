import unittest
import numpy as np
import torch

from pinn2d import dde, pde, boundary_lift, build_model


class CorrectedPINNTests(unittest.TestCase):
    def tearDown(self):
        dde.grad.clear()

    def test_real_deepxde_hessian_corrected(self):
        xy = torch.tensor([[.2, .3], [.7, .6]], requires_grad=True, dtype=torch.float64)
        x, y = xy[:, 0:1], xy[:, 1:2]
        u = x*x + 3*x*y + 2*y*y
        np.testing.assert_allclose(pde(xy, u).detach().numpy(), (6-.1*u).detach().numpy(), atol=1e-14)
        mixed = dde.grad.hessian(u, xy, component=0, i=1)
        np.testing.assert_allclose(mixed.detach().numpy(), 3., atol=1e-14)

    def test_all_open_edges_and_corner_convention(self):
        t = np.linspace(.0001, .9999, 61)
        points = np.vstack((np.column_stack((0*t, t)), np.column_stack((1+0*t, t)),
                            np.column_stack((t, 0*t)), np.column_stack((t, 1+0*t)),
                            [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]))
        xy = torch.tensor(points, dtype=torch.float64)
        values = boundary_lift(xy, torch.ones((len(points), 1), dtype=torch.float64)).detach().numpy().ravel()
        expected = np.r_[np.ones(len(t)), np.zeros(3*len(t)), [1., 1., 0., 0.]]
        np.testing.assert_allclose(values, expected, atol=5e-15)

    def test_lift_pde_derivatives_are_finite_near_corners(self):
        xy = torch.tensor([[1e-4, 2e-4], [1e-4, 1-2e-4], [.3, .4]], dtype=torch.float64, requires_grad=True)
        u = boundary_lift(xy, 0*xy[:, 0:1])
        values = pde(xy, u).detach().numpy()
        self.assertTrue(np.isfinite(values).all())
        self.assertLess(np.max(np.abs(values)), 10.)

    def test_lifting_is_not_an_exact_solution(self):
        xy = torch.tensor([[.3, .4], [.6, .2]], dtype=torch.float64, requires_grad=True)
        u = boundary_lift(xy, 0*xy[:, 0:1])
        self.assertGreater(float(torch.max(torch.abs(pde(xy, u))).detach()), .001)

    def test_lifting_laplacian_against_manual_derivative(self):
        xy = torch.tensor([[.1, .2], [.3, .4], [.6, .8], [1e-4, 2e-4]],
                          dtype=torch.float64, requires_grad=True)
        x, y = xy[:, 0:1], xy[:, 1:2]
        u = boundary_lift(xy, 0*x)
        # The two angle terms are harmonic. Differentiate the remaining terms.
        laplacian = (4*x/np.pi) * (-1/(1+x*x)**2 + y/(1+y*y)**2
                                   + (1-y)/(1+(1-y)**2)**2)
        expected = laplacian - .1*u
        np.testing.assert_allclose(pde(xy, u).detach().numpy(), expected.detach().numpy(), atol=1e-7, rtol=0)

    def test_modes_use_identical_interior_without_reference_labels(self):
        soft, a = build_model("soft", 0, n_interior=32, n_edge=8)
        lifted, b = build_model("lifted", 0, n_interior=32, n_edge=8)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all((a>0) & (a<1)))
        self.assertIsNone(soft.data.soln)
        self.assertIsNone(lifted.data.soln)


if __name__ == "__main__":
    unittest.main(verbosity=2)
