"""Checks of NEW reference/audit. Passing tests do NOT approve original PINN."""
import unittest
from pathlib import Path

import numpy as np

from reference import series_grid, finite_difference, reference_report, audit_notebook


class ReferenceTests(unittest.TestCase):
    def test_edges_and_explicit_corner_convention(self):
        points = np.linspace(0, 1, 21)
        values = series_grid(points, points)
        np.testing.assert_array_equal(values[:, 0], 1)
        np.testing.assert_array_equal(values[:, -1], 0)
        np.testing.assert_array_equal(values[0, 1:], 0)
        np.testing.assert_array_equal(values[-1, 1:], 0)

    def test_symmetry_and_concentration_bounds_away_from_corners(self):
        x = np.linspace(.025, .975, 25)
        y = np.linspace(.025, .975, 31)
        values = series_grid(x, y)
        np.testing.assert_allclose(values, values[::-1], atol=2e-14)
        self.assertGreater(values.min(), 0)
        self.assertLess(values.max(), 1)

    def test_laplace_zero_decay_center_by_symmetry(self):
        # Four rotated solutions sum to constant 1 when q=0; each is 1/4 at center.
        self.assertAlmostEqual(series_grid([.5], [.5], q=0)[0, 0], .25, places=13)

    def test_positive_decay_lowers_concentration(self):
        points = np.linspace(.1, .9, 17)
        self.assertTrue(np.all(series_grid(points, points, q=.1) < series_grid(points, points, q=0)))

    def test_independent_fd_manufactured_solution(self):
        errors = []
        q = .1
        for n in (16, 32, 64):
            coords, values, solver = finite_difference(
                n, q, source_left=False,
                forcing=lambda x, y: (2*np.pi**2 + q) * np.sin(np.pi*x) * np.sin(np.pi*y))
            exact = np.sin(np.pi*coords)[:, None] * np.sin(np.pi*coords)[None, :]
            errors.append(float(np.max(np.abs(values - exact))))
            self.assertLess(solver["relative_algebraic_residual"], 5e-11)
        self.assertTrue(all(3.9 < a/b < 4.1 for a, b in zip(errors, errors[1:])))

    def test_reference_residual_refinement_and_truncation(self):
        report = reference_report()
        self.assertLess(report["series_pde_residual_central_difference_h_1e-4_max_core"], 2e-5)
        self.assertTrue(all(1.8 < order < 2.2 for order in report["observed_core_orders"]))
        for row in report["refinement"]:
            self.assertLess(row["series_256_vs_512_max_on_interior"], 1e-7)
            self.assertGreaterEqual(row["fd_min"], -1e-11)
            self.assertLessEqual(row["fd_max"], 1+1e-11)

    def test_invalid_inputs(self):
        for call in (lambda: series_grid([-.1], [.5]), lambda: series_grid([.5], [np.nan]),
                     lambda: series_grid([.5], [.5], q=-1), lambda: finite_difference(2)):
            with self.assertRaises(ValueError):
                call()

    def test_original_is_flagged_not_silently_approved(self):
        notebook = Path(__file__).resolve().parent / "fixtures/original_PINN_2D_Biology.ipynb"
        report = audit_notebook(notebook)
        self.assertEqual(report["observed_pairs"], [[0, 0], [1, 0]])
        self.assertFalse(report["original_operator_matches_claimed_laplacian"])
        self.assertTrue(all(not call["j_explicit"] for call in report["hessian_calls"]))


class IndependentAutodiffTest(unittest.TestCase):
    def test_polynomial_separates_yx_from_yy(self):
        # Independent PyTorch Hessian, not a call to DeepXDE or original code.
        import torch
        point = torch.tensor([.2, .3], dtype=torch.float64, requires_grad=True)
        hessian = torch.autograd.functional.hessian(lambda z: z[0]**2 + 3*z[0]*z[1] + 2*z[1]**2, point)
        np.testing.assert_allclose(hessian.detach().numpy(), [[2, 3], [3, 4]], atol=1e-14)
        self.assertEqual(float(hessian[0, 0] + hessian[1, 0]), 5.)
        self.assertEqual(float(hessian[0, 0] + hessian[1, 1]), 6.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
