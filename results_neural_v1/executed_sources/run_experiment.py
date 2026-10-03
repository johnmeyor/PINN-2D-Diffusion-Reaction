"""Run/reload corrected PINN experiments. Does not execute the original notebook."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pinn2d import dde, build_model, evaluate


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for stream in self.streams: stream.write(text); stream.flush()
    def flush(self):
        for stream in self.streams: stream.flush()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def plot_result(path, coords, prediction, ref, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, values, title in zip(axes, (ref, prediction, np.abs(prediction-ref)),
                                 ("Independent reference", label, "Absolute error")):
        im = ax.imshow(values, origin="lower", extent=[0, 1, 0, 1],
                       vmin=0, vmax=(1 if title != "Absolute error" else None), cmap="viridis")
        ax.set(title=title, xlabel="x", ylabel="y")
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def train_run(root, mode, seed, adam, lbfgs):
    folder = root / f"{mode}_seed{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    config = {"mode": mode, "seed": seed, "n_interior": 1024, "n_edge": 128,
              "width": 32, "hidden_layers": 3, "adam_iterations": adam,
              "lbfgs_maxiter": lbfgs, "dtype": "float64", "backend": "pytorch",
              "pde": "c_xx+c_yy-0.1*c=0", "revision_run": True}
    write_json(folder / "config.json", config)
    with (folder / "training.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)):
            model, interior = build_model(mode, seed)
            np.save(folder / "training_interior.npy", interior)
            weights = [1] if mode == "lifted" else [1, 10, 10]
            started = time.perf_counter()
            model.compile("adam", lr=1e-3, loss_weights=weights)
            model.train(iterations=adam, display_every=400)
            dde.optimizers.config.set_LBFGS_options(maxiter=lbfgs, maxfun=int(lbfgs*1.5),
                                                  ftol=1e-12, gtol=1e-8, maxls=30)
            # Bound each internal PyTorch call, including short smoke experiments.
            dde.optimizers.config.LBFGS_options["iter_per_step"] = min(100, lbfgs)
            dde.optimizers.config.LBFGS_options["fun_per_step"] = min(150, int(lbfgs*1.5))
            model.compile("L-BFGS", loss_weights=weights)
            model.train(display_every=100)
            elapsed = time.perf_counter() - started
            metrics, coords, prediction, ref = evaluate(model)
            metrics["training_seconds"] = elapsed
            metrics["final_step"] = model.train_state.step
            torch.save(model.net.state_dict(), folder / "weights.pt")
            np.savez_compressed(folder / "fields.npz", coords=coords, prediction=prediction, reference=ref)
            write_json(folder / "metrics.json", metrics)
            plot_result(folder / "comparison.png", coords, prediction, ref, f"{mode}, seed {seed}")
            print(json.dumps({"mode": mode, "seed": seed, "metrics": metrics}, indent=2), flush=True)
    return {"mode": mode, "seed": seed, **metrics}


def audit_run(folder):
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    model, _ = build_model(config["mode"], config["seed"])
    model.compile("adam", lr=1e-3, loss_weights=([1] if config["mode"] == "lifted" else [1, 10, 10]))
    model.net.load_state_dict(torch.load(folder / "weights.pt", weights_only=True, map_location="cpu"))
    metrics, _, prediction, ref = evaluate(model)
    saved = np.load(folder / "fields.npz")
    stored_metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    difference = float(np.max(np.abs(prediction - saved["prediction"])))
    reference_difference = float(np.max(np.abs(ref - saved["reference"])))
    passed = difference <= 1e-12 and reference_difference <= 1e-12
    passed = passed and metrics["preregistered_criteria"] == stored_metrics["preregistered_criteria"]
    report = {"weights_sha256": hashlib.sha256((folder / "weights.pt").read_bytes()).hexdigest(),
              "reloaded_prediction_max_difference": difference,
              "reference_max_difference": reference_difference,
              "reloaded_metrics": metrics, "audit_pass": passed}
    write_json(folder / "reload_audit.json", report)
    if not passed: raise RuntimeError("Reload audit failed")
    return report


def analytic_baseline(root):
    """Measure the boundary lifting alone, so its benefit is not credited to ML."""
    folder = root / "analytic_lifting_only"
    folder.mkdir(parents=True, exist_ok=False)
    model, _ = build_model("lifted", 0)
    model.compile("adam", lr=1e-3)
    with torch.no_grad():
        for parameter in model.net.parameters():
            parameter.zero_()
    metrics, coords, prediction, ref = evaluate(model)
    write_json(folder / "metrics.json", metrics)
    np.savez_compressed(folder / "fields.npz", coords=coords, prediction=prediction, reference=ref)
    plot_result(folder / "comparison.png", coords, prediction, ref, "Analytic lifting only; no ML")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--modes", nargs="+", choices=["soft", "lifted"], default=["soft", "lifted"])
    parser.add_argument("--adam", type=int, default=1200)
    parser.add_argument("--lbfgs", type=int, default=300)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        for mode in args.modes:
            for seed in args.seeds:
                report = audit_run(args.output / f"{mode}_seed{seed}")
                print(mode, seed, "reload audit:", report["audit_pass"], flush=True)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        source_names = ("pinn2d.py", "reference.py", "run_experiment.py", "PROTOCOL.md")
        write_json(args.output / "provenance.json", {
            "python": platform.python_version(),
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "deepxde", "numpy", "scipy", "matplotlib")},
            "source_sha256": {p: hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest() for p in source_names},
            "argv": sys.argv, "scope": "Corrected extension runs; not a reproduction of the original architecture, backend or output"})
        analytic_baseline(args.output)
        rows = []
        for mode in args.modes:
            for seed in args.seeds:
                rows.append(train_run(args.output, mode, seed, args.adam, args.lbfgs))
                write_json(args.output / "summary.json", rows)
