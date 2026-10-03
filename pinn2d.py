"""Corrected DeepXDE PINN and boundary-aware variant.

Same square, equation and open-edge boundary values as the source notebook.
No series/reference values enter either training loss.
"""
import os
os.environ["DDE_BACKEND"] = "pytorch"  # before import; no global config file change
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import deepxde as dde
import numpy as np
import torch

from reference import series_grid

dde.config.set_default_float("float64")
torch.set_num_threads(2)


def pde(x, c):
    c_xx = dde.grad.hessian(c, x, component=0, i=0, j=0)
    c_yy = dde.grad.hessian(c, x, component=0, i=1, j=1)
    return c_xx + c_yy - 0.1 * c


def boundary_lift(xy, raw):
    """Analytic corner lifting + homogeneous neural correction, not exact solution.

    Angle terms handle the two incompatible corner limits without smoothing the
    specified boundary data. See PROTOCOL.md for derivation and limitations.
    """
    x, y = xy[:, 0:1], xy[:, 1:2]
    right_trace = (2 / np.pi) * (torch.atan(y) + torch.atan(1-y)) - 0.5
    lift = (2 / np.pi) * (torch.atan2(y, x) + torch.atan2(1-y, x) + torch.atan(x)) - 1 - x * right_trace
    result = lift + x * (1-x) * y * (1-y) * raw
    # Exact isolated-corner convention. Never differentiate at boundary points.
    return torch.where(x == 0, torch.ones_like(result), result)


def build_model(mode, seed, n_interior=1024, n_edge=128, width=32):
    if mode not in ("soft", "lifted"):
        raise ValueError("mode must be soft or lifted")
    dde.config.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    interior = rng.uniform(np.finfo(float).eps, 1-np.finfo(float).eps, (n_interior, 2))
    t = (np.arange(n_edge) + 0.5) / n_edge
    left = np.column_stack((np.zeros_like(t), t))
    other = np.vstack((np.column_stack((np.ones_like(t), t)),
                       np.column_stack((t, np.zeros_like(t))),
                       np.column_stack((t, np.ones_like(t)))))
    boundaries = [] if mode == "lifted" else [
        dde.icbc.PointSetBC(left, np.ones((len(left), 1))),
        dde.icbc.PointSetBC(other, np.zeros((len(other), 1)))]
    geometry = dde.geometry.Rectangle([0, 0], [1, 1])
    data = dde.data.PDE(geometry, pde, boundaries, num_domain=0, num_boundary=0, anchors=interior)
    net = dde.nn.FNN([2, width, width, width, 1], "tanh", "Glorot normal")
    net.apply_feature_transform(lambda x: 2*x-1)
    if mode == "lifted":
        net.apply_output_transform(boundary_lift)
    model = dde.Model(data, net)
    return model, interior


def error_metrics(prediction, reference):
    difference = np.asarray(prediction).ravel() - np.asarray(reference).ravel()
    return {"relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
            "max_absolute": float(np.max(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference**2)))}


def evaluate(model):
    coords = np.linspace(0, 1, 129)
    xx, yy = np.meshgrid(coords, coords)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    prediction = model.predict(points).reshape(xx.shape)
    ref = series_grid(coords, coords, modes=512)
    interior = (xx > 0) & (xx < 1) & (yy > 0) & (yy < 1)
    core = (xx >= .125) & (xx <= .875) & (yy >= .125) & (yy <= .875)
    corner_distance = np.minimum(np.hypot(xx, yy), np.hypot(xx, 1-yy))
    near = interior & (corner_distance < .125)
    metrics = {"interior": error_metrics(prediction[interior], ref[interior]),
               "core": error_metrics(prediction[core], ref[core]),
               "near_left_corners": error_metrics(prediction[near], ref[near]),
               "predicted_min": float(prediction.min()), "predicted_max": float(prediction.max())}
    t = np.linspace(.005, .995, 199)
    edges = {"left": np.column_stack((t*0, t)), "right": np.column_stack((t*0+1, t)),
             "bottom": np.column_stack((t, t*0)), "top": np.column_stack((t, t*0+1))}
    metrics["edge_max_errors_excluding_corners"] = {
        name: float(np.max(np.abs(model.predict(pts).ravel() - (1 if name == "left" else 0))))
        for name, pts in edges.items()}
    corners = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    metrics["corner_values_source_priority"] = model.predict(corners).ravel().tolist()
    test = np.linspace(.031, .969, 25)
    rx, ry = np.meshgrid(test, test)
    residual = model.predict(np.column_stack((rx.ravel(), ry.ravel())), operator=pde)
    metrics["pde_test_rmse"] = float(np.sqrt(np.mean(residual**2)))
    metrics["pde_test_max"] = float(np.max(np.abs(residual)))
    criteria = {"interior_relative_l2_le_5pct": metrics["interior"]["relative_l2"] <= .05,
                "interior_max_absolute_le_005": metrics["interior"]["max_absolute"] <= .05,
                "core_max_absolute_le_002": metrics["core"]["max_absolute"] <= .02,
                "open_edge_max_le_002": max(metrics["edge_max_errors_excluding_corners"].values()) <= .02,
                "pde_test_rmse_le_002": metrics["pde_test_rmse"] <= .02,
                "range_violation_le_002": metrics["predicted_min"] >= -.02 and metrics["predicted_max"] <= 1.02}
    metrics["preregistered_criteria"] = criteria
    metrics["all_criteria_pass"] = all(criteria.values())
    if not np.isfinite(prediction).all() or not np.isfinite(residual).all():
        raise RuntimeError("Non-finite prediction or residual")
    return metrics, coords, prediction, ref
