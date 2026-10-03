# Protocol fixed before new training, 17 September 2026

This protocol records the corrected and extended September 2026 experiment
before training. It is distinct from the preserved historical notebook.

## What is fixed and what changes

Keep the original equation `c_xx+c_yy-0.1c=0` on the unit square. Left open edge:
1; other three open edges: 0. Do not smooth boundary data or replace the task.
Fix the Hessian indices to `(0,0)` and `(1,1)`. Select PyTorch before importing
DeepXDE, in a new isolated environment; do not modify a global backend setting.

Compare two corrected models: a plain soft-Dirichlet PINN and a corner-aware
lifting with a neural correction. Both use DeepXDE, float64, `tanh`, three hidden
layers of width32, feature normalization `2x-1`, seeds0,1,2 and the same1024
pseudorandom interior coordinates per seed. The soft model also uses128 midpoint
points per edge, excluding the discontinuous exact corners; weights `[1,10,10]`
for PDE, left and other edges. The lifted model uses PDE loss only because its
open-edge conditions hold by construction. No series values enter either loss.

Training budget: Adam1200 at0.001, then at most300 L-BFGS iterations per run.
Report actual steps and elapsed times. This is not the original128x5 network,
TensorFlow environment or10000-iteration training. It is a bounded comparison of
two new corrected implementations. Do not claim equal compute from equal steps.
No tuning on the reported evaluation grid after viewing final results. A very
short smoke run checks execution only and is excluded from accuracy claims.

Additional control fixed before the six full runs: evaluate the analytical
lifting alone with the neural correction set to zero. This separates the
contribution of the mathematical lifting from the benefit of neural training;
do not credit the network with accuracy already supplied by the lifting.

## Boundary lifting, not a hidden exact solution

For interior x>0, let

```
R(y) = (2/pi)*(atan(y)+atan(1-y)) - 1/2
A(x,y) = (2/pi)*(atan(y/x)+atan((1-y)/x)+atan(x)) - 1 - x*R(y)
c_theta(x,y) = A(x,y) + x*(1-x)*y*(1-y)*N_theta(x,y).
```

The code uses `atan2` to evaluate the angles. Each angle associated with a corner
is harmonic away from that corner. The remaining terms make the three other
open edges zero. Direct substitution gives A=1 on the open left edge, A=0 on
the other open edges. The polynomial neural factor vanishes on every edge.
A generally has a nonzero PDE residual: it is not the reference solution and
does not incorporate series/test-grid labels. The discontinuous corner behavior
is represented analytically, not approximated by requiring a continuous `tanh`
network to have contradictory limits. The isolated left-corner value is1, as in
the original selector; no derivatives are evaluated at a corner/boundary point.

This is an application of analytical boundary lifting, not a claim of a new
PINN method. Compare measured accuracy, boundary errors and wall time openly.

## Independent evaluation and pass criteria

Use a129x129 grid, separately all interior points and core `[1/8,7/8]^2`; report
the near-left-corner interior subset with radius1/8 as well. Reference:512 odd
sine modes from `reference.py`, already cross-checked against finite differences.
Report relative L2, RMSE and absolute maximum; never substitute training loss
for solution error. Test all four edges at199 points from0.005 to0.995 and show
corner values separately. Report min/max concentration and PDE residual on a
different25x25 grid from0.031 to0.969.

Predefined diagnostic thresholds (not universal scientific certification):

- all-interior relative L2 <=5%; all-interior absolute maximum <=0.05;
- core absolute maximum <=0.02;
- maximum open-edge error <=0.02;
- test PDE residual RMSE <=0.02;
- range violation outside[0,1] <=0.02.

Publish every seed/run result, including failures. No physical/biological
validation follows from passing these sampled checks. Sharp corner limits and
continuum error bounds are not certified by finite grids.

## Artifacts and audit

Keep source original unchanged. Save config, training coordinates/logs, weights,
reference/predicted arrays, metrics, figures and source hashes for every run.
Reload each checkpoint in a separate audit invocation and compare predictions,
reference and pass/fail criteria. Retain both old and new outputs with labels.
No GitHub publication, portal upload or CV claims without John reviewing the
work and authorizing the relevant external action.
