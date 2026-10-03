"""Audit/reference implementation, not execution of the original PINN.

NumPy-only mathematical reference for D*Laplacian(c)-k*c=0 on the unit square.
Rows index y, columns index x. Original notebook is parsed, NEVER executed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

ORIGINAL_SHA256 = "c130aade83923ac7ba3eb18fab1c393c008e160a58385fb61552b0047eb377fc"
ORIGINAL_COMMIT = "5cfcce750f2c5f91b8172c4301c4e517faf9f0e7"


def series_grid(x, y, q=0.1, modes=256):
    """Odd sine expansion; q=k/D. Corner values follow source-priority convention.

    c(0,y)=1 including corners; c(x,0)=c(x,1)=0 for x>0. This is NOT a
    continuous trace at the two left corners. Main comparisons exclude boundaries.
    'modes' counts odd modes (largest index 2*modes-1).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("Finite one-dimensional coordinate arrays required")
    if np.any((x < 0) | (x > 1)) or np.any((y < 0) | (y > 1)):
        raise ValueError("Coordinates must lie in [0,1]")
    if not np.isfinite(q) or q < 0 or not isinstance(modes, int) or modes < 1:
        raise ValueError("q>=0 and positive integer modes required")
    n = np.arange(1, 2 * modes, 2, dtype=float)
    alpha = np.sqrt((n * np.pi) ** 2 + q)
    # Stable sinh(alpha*(1-x))/sinh(alpha), with no overflowing sinh(alpha).
    ratio = (np.exp(-alpha[:, None] * x[None, :])
             * (-np.expm1(-2 * alpha[:, None] * (1 - x[None, :])))
             / (-np.expm1(-2 * alpha[:, None])))
    result = np.sin(np.pi * y[:, None] * n[None, :]) @ ((4 / (n * np.pi))[:, None] * ratio)
    result[(y == 0) | (y == 1), :] = 0
    result[:, x == 1] = 0
    result[:, x == 0] = 1  # isolated corner convention, not uniform convergence
    return result


def stencil(values, h, q=0.1):
    """Interior five-point operator -Laplacian(c)+q*c with zero ghost boundary."""
    out = (4 + q * h * h) * values.copy()
    out[1:, :] -= values[:-1, :]
    out[:-1, :] -= values[1:, :]
    out[:, 1:] -= values[:, :-1]
    out[:, :-1] -= values[:, 1:]
    return out


def finite_difference(n, q=0.1, *, source_left=True, forcing=None, rtol=1e-12):
    """Independent five-point FD solution by conjugate gradients; n intervals.

    Solves -Delta(c)+q*c=f, for constant left boundary 1 or all-zero boundaries.
    Returns coordinates, field, and measured relative algebraic residual.
    """
    if not isinstance(n, int) or n < 4 or not np.isfinite(q) or q < 0:
        raise ValueError("n>=4 and q>=0 required")
    h = 1 / n
    coords = np.linspace(0, 1, n + 1)
    xx, yy = np.meshgrid(coords[1:-1], coords[1:-1])
    rhs = np.zeros_like(xx) if forcing is None else np.asarray(forcing(xx, yy)) * h * h
    if rhs.shape != xx.shape or np.any(~np.isfinite(rhs)):
        raise ValueError("Forcing must return a finite interior-grid array")
    if source_left:
        rhs[:, 0] += 1
    u = np.zeros_like(rhs)
    residual = rhs.copy()
    direction = residual.copy()
    rr = float(np.sum(residual * residual))
    norm_rhs = np.sqrt(rr)
    iterations = 0
    for iterations in range(1, 10 * n + 1):
        if np.sqrt(rr) <= rtol * max(norm_rhs, 1e-300):
            break
        applied = stencil(direction, h, q)
        step = rr / float(np.sum(direction * applied))
        u += step * direction
        residual -= step * applied
        next_rr = float(np.sum(residual * residual))
        direction = residual + (next_rr / rr) * direction
        rr = next_rr
    actual_residual = float(np.linalg.norm(stencil(u, h, q) - rhs) / max(norm_rhs, 1e-300))
    if actual_residual > 5e-11:
        raise RuntimeError(f"FD algebraic residual too large: {actual_residual}")
    field = np.zeros((n + 1, n + 1))
    field[1:-1, 1:-1] = u
    if source_left:
        field[:, 0] = 1
    return coords, field, {"iterations": iterations, "relative_algebraic_residual": actual_residual}


def audit_notebook(path):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != ORIGINAL_SHA256:
        raise ValueError("Source differs from reviewed notebook; re-inspect before audit")
    notebook = json.loads(raw)
    matches = []
    for cell_index, cell in enumerate(notebook["cells"]):
        text = "".join(cell.get("source", []))
        if cell.get("cell_type") != "code" or "def pde(" not in text:
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "dde.grad.hessian":
                keywords = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
                matches.append({"cell_index": cell_index, "line_in_cell": node.lineno,
                                "i": keywords.get("i", 0), "j": keywords.get("j", 0),
                                "j_explicit": "j" in keywords})
    pairs = [[row["i"], row["j"]] for row in matches]
    return {"sha256": digest, "commit": ORIGINAL_COMMIT, "hessian_calls": matches,
            "documented_default_j": 0, "expected_laplacian_pairs": [[0, 0], [1, 1]],
            "observed_pairs": pairs, "original_operator_matches_claimed_laplacian": pairs == [[0, 0], [1, 1]],
            "basis": "Static parsing plus official DeepXDE documented API; original not executed",
            "api_source": "https://deepxde.readthedocs.io/en/v1.13.1/modules/deepxde.gradients.html"}


def reference_report():
    rows = []
    for n in (32, 64, 128):
        coords, fd, solver = finite_difference(n)
        reference = series_grid(coords, coords, modes=512)
        xx, yy = np.meshgrid(coords, coords)
        interior = (xx > 0) & (xx < 1) & (yy > 0) & (yy < 1)
        core = (xx >= 0.125) & (xx <= 0.875) & (yy >= 0.125) & (yy <= 0.875)
        difference = fd - reference
        truncation = np.max(np.abs(reference[interior] - series_grid(coords, coords, modes=256)[interior]))
        rows.append({"n_intervals": n, **solver,
                     "all_interior_relative_l2": float(np.linalg.norm(difference[interior]) / np.linalg.norm(reference[interior])),
                     "all_interior_max_absolute": float(np.max(np.abs(difference[interior]))),
                     "core_max_absolute": float(np.max(np.abs(difference[core]))),
                     "series_256_vs_512_max_on_interior": float(truncation),
                     "fd_min": float(fd.min()), "fd_max": float(fd.max())})
    orders = [float(np.log2(a["core_max_absolute"] / b["core_max_absolute"]))
              for a, b in zip(rows, rows[1:])]
    grid = np.linspace(0.125, 0.875, 13)
    h = 1e-4
    center = series_grid(grid, grid)
    residual = ((series_grid(grid+h, grid) - 2*center + series_grid(grid-h, grid)) / h**2
                + (series_grid(grid, grid+h) - 2*center + series_grid(grid, grid-h)) / h**2 - .1*center)
    return {"problem": "Delta(c)-0.1*c=0; left=1; other open edges=0; unit square",
            "method": "odd sine series vs independent five-point finite differences/CG",
            "core_region": "[1/8,7/8]^2, excludes both corner singularities",
            "refinement": rows, "observed_core_orders": orders,
            "series_pde_residual_central_difference_h_1e-4_max_core": float(np.max(np.abs(residual))),
            "center_concentration": float(series_grid([.5], [.5])[0, 0]),
            "neural_prediction_errors": None,
            "limitation": "No original network weights or predictions evaluated; no retraining; no biological validation"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"environment": {"python": platform.python_version(), "numpy": np.__version__},
              "audit": audit_notebook(args.notebook), "reference": reference_report(),
              "scope": "New audit/reference implementation; not execution of the original PINN"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
