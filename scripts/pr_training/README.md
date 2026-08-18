# `pos_geo_training.py` — Position-Geometry Training

Trains a classifier on perturbations that sit on a **margin level set**, and
records where on that level set each perturbation landed.

The question it exists to answer: *at the same margin level, does it matter
which positions the training signal comes from?*

## What one training step does

1. Draw `N` random perturbations inside the ε-ball (`--num_starts`).
2. Drive each one onto the level set `m(x + δ, y) = t` (`--t`), by minimising
   a symmetric penalty on `m − t` plus an L2 anchor to its own random start.
3. Score every landing position and turn the scores into weights (`--weight_mode`).
4. Train on the weighted mixture: `loss = Σ_r w_r · CE(f(x + δ_r), y)`.

Both `δ` and `w` are detached — the gradient flows only into the model.

## The five weighting strategies

| `--weight_mode` | Weight | Reads as |
|---|---|---|
| `uniform` | equal over valid positions | average all δ |
| `sharp` | `w ∝ exp(+g/τ)` | favour steep spots |
| `flat` | `w ∝ exp(−g/τ)` | favour wide-valley spots |
| `min_norm` | `1` on `argmin ‖δ‖`, else `0` | only the closest perturbation |
| `max_norm` | `1` on `argmax ‖δ‖`, else `0` | only the furthest perturbation |

`g = ‖∇_δ CE‖_*` is the **dual norm of the cross-entropy input gradient** at
the landing point — L1 for an L∞ ball, L2 for an L2 ball. It measures local
steepness of the *loss surface*, which is what "sharp minimum" and "wide
valley" refer to. `--tau` is the softmax temperature; large values flatten
`sharp` / `flat` back toward `uniform`.

**Invalid positions get zero weight in every mode.** A perturbation that never
reached the level set (`|m − t| > tol`) is not on the level being studied, so
it is excluded and the remaining weights renormalise. If *all* `N` miss, that
image contributes no gradient that step — watch `valid_rate`.

## Outputs

```
ckp/pos_geo_training/<tag>.pth                    checkpoint (+ level_config)
ckp/pos_geo_training/<tag>.log                    training log
results/pos_geo_training/<tag>_training_info.csv  per-epoch summary
results/pos_geo_training/<tag>_probe_ep{N}.npz    per-delta record
```

### `_training_info.csv` — one row per epoch

Geometry columns are filled **every** epoch. Evaluation is far more expensive,
so it runs every 5 epochs (and on the last); its columns are empty otherwise,
and the schema is identical across rows.

| Group | Columns |
|---|---|
| Run | `arch` `dataset` `training_type` `weight_mode` `epoch` `lr` `time` |
| Training | `train_loss` `train_acc` |
| Evaluation | `trainS_loss` `trainS_acc` `trainS_pgd` `val_loss` `val_acc` `val_pgd` |
| Margins | `margin1` `margin2` `margin_gap` `ce` |
| Positions | `delta_norm` `grad_dual` `valid_rate` |
| Dispersion | `pair_cos` `eff_rank` `sigma_max` `sigma_min` `anisotropy` |
| Weights | `weight_entropy` |

Evaluation is deliberately narrow: clean accuracy and PGD-10 robust accuracy,
on the train subset and the test set. Nothing else.

### `_probe_ep{N}.npz` — per-delta detail

A **fixed** probe subset (`--probe_n`, default 256) is re-measured every epoch,
so one image's perturbations can be tracked as training proceeds. With
`R = probe_n` and `N = num_starts`:

| Key | Shape | Meaning |
|---|---|---|
| `delta_norm` | `(R, N)` | ‖δ‖₂ of each perturbation |
| `margin1` / `margin2` | `(R, N)` | margin against the top-1 / top-2 wrong class |
| `j1` / `j2` | `(R, N)` | which classes those are |
| `ce` | `(R, N)` | cross-entropy at each position |
| `grad_dual` | `(R, N)` | `‖∇_δ CE‖_*` — local sharpness |
| `weights` | `(R, N)` | the weights actually used |
| `valid_mask` | `(R, N)` | reached the level set? |
| `pair_cos` | `(R, N, N)` | pairwise cosine between perturbations |
| `eff_rank` `sigma_max` `sigma_min` `anisotropy` | `(R,)` | dispersion, see below |
| `indices` | `(R,)` | which training images these are (constant across epochs) |

The probe never changes the model: it forwards in eval mode so BatchNorm
running statistics are untouched.

## Reading the diagnostics

**`margin_gap = margin2 − margin1`** — how firmly one wrong class owns the
decision. A small gap means the runner-up can overtake between steps, so that
position's "opponent" is unstable.

**`eff_rank`** — the participation ratio of the pairwise-cosine spectrum,
`(Σσ)² / Σσ²`, in `[1, N]`. It reads directly as *how many independent
directions the N perturbations span*:

- `≈ 1` → they collapsed onto one direction; the N positions are not really different
- `≈ N` → mutually orthogonal, maximally spread

Computed from `eigvalsh` rather than an SVD — the cosine matrix is a symmetric
PSD Gram matrix, so its eigenvalues *are* its singular values, and unlike a
condition number `eff_rank` stays finite when the perturbations go collinear.

**`weight_entropy`** — Shannon entropy of each weight row: `log(N)` when all
positions share equally, `0` when one takes everything. `sharp` / `flat` can
saturate and collapse onto a single δ, which makes the weighting meaningless;
this column is how you catch it.

**`valid_rate`** — the fraction of perturbations that reached the level set. If
it is low, the solver is not converging: raise `--num_steps`, raise
`--epsilon`, or lower `--anchor_lambda`. Everything downstream is conditional
on this being healthy.

## Key hyper-parameters

| Flag | Default | Notes |
|---|---|---|
| `--t` | `0.0` | target margin level; `0` is the decision boundary |
| `--num_starts` | `8` | N positions per image; also the max `eff_rank` |
| `--num_steps` | `50` | solver steps; too few and `valid_rate` collapses |
| `--anchor_lambda` | `0.02` | **keep small.** Large values stop perturbations reaching the level set |
| `--tol` | `0.05` | `\|m − t\| ≤ tol` counts as valid |
| `--weight_mode` | `uniform` | the experiment |
| `--tau` | `1.0` | temperature for `sharp` / `flat` |
| `--probe_n` | `256` | probe subset size |

On the anchor, measured rather than assumed: in CIFAR's 3072 input dimensions
random starts are already ~19 apart and the solver moves each only ~0.6, so the
perturbations never collapse together even at `anchor_lambda = 0`. What the
anchor actually trades off is drift-from-start against reaching the level at
all — `λ ≥ 0.2` was enough to stop most perturbations landing within `tol`.

## Running it

```bash
# Single run
python scripts/pr_training/pos_geo_training.py \
    --dataset cifar10 --arch resnet18 --training_type level \
    --t 0.0 --num_starts 8 --num_steps 50 --weight_mode flat \
    --eval_pgd --pgd_steps 10 --epochs 100

# Sweep all five strategies on one arch/dataset
bash run_exp/pos_geo_training/run_pos_geo.sh

# Full arch x dataset x weight_mode grid
bash run_exp/pos_geo_training/run_pos_geo_sweep.sh
```

The shell scripts rename each run's outputs to
`pos_geo_<mode>_t<T>_N<NUM_STARTS>.*` so a sweep does not clobber itself.

`--training_type standard` runs plain clean-image training as a control; it
skips the solver and writes no geometry.

## Where the code lives

| File | Role |
|---|---|
| `scripts/pr_training/pos_geo_training.py` | training loop, logging, CLI |
| `src/pos_geo_loss.py` | solver, weighting, geometry |
| `utils/epoch_eval.py` | clean + PGD evaluation |
| `run_exp/pos_geo_training/` | sweep scripts |

`src/pos_geo_loss.py` is where the maths is: `solve_level_perturbations`
(drive δ onto the level set), `pos_geo_weights` (the five strategies),
`pos_geo_loss` (the weighted objective), `level_geometry` + `dispersion` (the
position measurements).
