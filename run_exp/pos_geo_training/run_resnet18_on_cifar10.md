# ResNet-18 / CIFAR-10 — level-set position-geometry ablation

Companion to [`run_resnet18_on_cifar10.sh`](./run_resnet18_on_cifar10.sh).

The solver and weighting in [`src/pos_geo_loss.py`](../../src/pos_geo_loss.py) are
debugged and unit-tested, and short runs behave: `t_mode=reachable` with
`solver=newton` holds `ae_rate_valid = 1.000` and a valid rate of 0.927 → 0.876
over six epochs, against 0.827 → 0.162 for a fixed `t = −0.5`. But nothing has
been trained to convergence, so **"the method works" is still a hypothesis.**
This experiment is what tests it.

## What this experiment answers

Five questions, each with a stated way of coming out negative. The design is
one-factor-at-a-time: every run differs from `ref` in exactly one thing.

| Question | Axis | What would falsify it |
|---|---|---|
| At the same margin level, does it matter *which positions* the training signal comes from? This is why the method exists. | `weight_mode` × 5 | the five modes land within noise of each other |
| Does newton's convergence advantage become robustness, or does it only make the valid rate look good? | `solver` × 2 | `energy` matches it on test PGD |
| Is `reachable` actually better than a fixed level, and how strong should training be? | `t_frac` × 3, `t_mode=fixed` × 2 | a fixed `t` does just as well |
| **Do multiple starts do anything at all?** | `num_starts` 1 / 4 / 8 | `starts1` ties with `ref` |
| Should sharp/flat score by L2 or by the dual norm? | `geometry_mode` × 2 | no measurable difference |

The `num_starts=1` control is the sharpest of these. With one start every
`weight_mode` collapses to the same computation, so if it ties with `ref` then
the weighting axis is answering a question that does not exist, and the right
response is to revisit the method rather than to run the expensive stage.

## How to run

```bash
cd ~/nppr && export PYTHONPATH=~/nppr

# 1. Screen: all 14 ablation runs at cheap solver settings
bash run_exp/pos_geo_training/run_resnet18_on_cifar10.sh screen

# 2. Baselines: standard / PGD-AT / FGSM-RS, same optimiser and epochs
bash run_exp/pos_geo_training/run_resnet18_on_cifar10.sh baseline

# 3. Summarise, and splice the table into this file
python scripts/pr_training/summarize_ablation.py \
    results/pos_geo_training/resnet18_cifar10/screen \
    --out run_exp/pos_geo_training/run_resnet18_on_cifar10.md

# 4. Put the winners in CONFIRM_TAGS at the top of the .sh, then
bash run_exp/pos_geo_training/run_resnet18_on_cifar10.sh confirm
```

Both GPUs, by sharding rather than queueing — each process takes every
`NUM_SHARDS`-th run:

```bash
GPU_ID=0 SHARD=0 NUM_SHARDS=2 bash run_exp/pos_geo_training/run_resnet18_on_cifar10.sh screen &
GPU_ID=1 SHARD=1 NUM_SHARDS=2 bash run_exp/pos_geo_training/run_resnet18_on_cifar10.sh screen &
```

### Cost

Solver cost is roughly linear in `num_starts × num_steps`. Measured on this box:
**48 s/epoch** at `4 × 10`, batch size 256.

| Stage | Settings | Per run | Total | Wall clock, 2×3090 |
|---|---|--:|--:|--:|
| `screen` | `num_starts=4 num_steps=20 epochs=30` | ~48 min | 14 runs ≈ 12 GPU-h | ~6 h |
| `baseline` | same epochs | ~10 min | 3 runs | <1 h |
| `confirm` | `num_starts=8 num_steps=50 epochs=100` | ~13 h | top 3 + baselines | ~20 h |

A full grid over the same five axes would be 120 runs at ~13 h each. That is
what the two-stage design buys.

Nothing is written to `ckp/` — the CSVs are the product. Flip
`KEEP_BASELINE_CKPT=1` in the script to keep the three baseline checkpoints,
which are the reference models a later AutoAttack comparison would want.

## The reference configuration

Everything below is held fixed; each ablation run changes one line of it.

```
arch=resnet18  dataset=cifar10  bs=1024  lr=0.01  wd=5e-4  seed=42  augment=off
norm=linf  epsilon=auto(8/255)
solver=newton       t_mode=reachable  t_frac=0.5
weight_mode=uniform geometry_mode=coarea
tol_mode=absolute   tol=auto(0.05)
step_size=auto(1.0) anchor_lambda=auto(0)
```

Why each of these, with the measurement behind it:

| Choice | Evidence |
|---|---|
| `solver=newton` | Scale free — no `psi_alpha` to retune as the margin scale moves during training. Valid rate 0.22 → 0.62 on a trained model at ten steps, same cost per step. |
| `anchor_lambda` auto (**0** for newton) | A Newton step shrinks to nothing near the level set while a fixed anchor pull does not, so the anchor eventually dominates and drags the solver back off. Valid rate 0.228 → 0.898 and median &#124;m−t&#124; 0.20 → 0.003 with it removed. `energy` keeps 0.01; its gradient stays O(1) in the saturated region, so the same argument does not apply. |
| `step_size` auto | 1.0 for newton (the full Newton step), 0.2 for energy. A shared value silently mis-tunes one of them; for energy, 1e-2 → 0.2 moved the valid rate 0.24 → 0.84. |
| `t_mode=reachable`, `t_frac=0.5` | The reachable margin floor rises as the model hardens — measured −1.54 → −0.20 over six epochs — so a fixed level walks out of the ε-ball and the valid rate went 0.96 → 0.15. `t_frac=0.5` held 0.88–1.00 over the same runs. Costs five extra backward passes per batch. |
| `tol_mode=absolute` | With `anchor_lambda=0` the solver converges to &#124;m−t&#124; ≈ 1e-4, so a fixed `tol=0.05` is no longer the binding constraint and the relative mode's extra machinery buys nothing here. |
| `geometry_mode=coarea` | See below — it is a theory question, not a measured one. |

### Why `coarea` (L2) and not `dual`, under an L∞ threat model

The coarea formula is an identity of **Euclidean** geometric measure theory:

$$\int_B g(\delta)\,\lVert\nabla m(\delta)\rVert_2\,d\delta \;=\; \int_{\mathbb{R}}\left(\int_{\{m=t\}} g\;d\mathcal{H}^{d-1}\right)dt$$

$\mathcal{H}^{d-1}$ is Euclidean surface measure, and the $\lVert\nabla m\rVert_2$
inside is **not a free choice** — substitute $\lVert\nabla m\rVert_1$ and the
identity is simply false. So as long as the weight claims to approximate the PR
surface measure — which is exactly what
[`pos_geo_loss.py`](../../src/pos_geo_loss.py) documents as
`w(δ) ∝ q(δ) / ‖∇_δ m(δ)‖₂` — the factor has to be L2, whether the threat set is
an L∞ ball or an L2 ball.

**The threat norm enters this theory somewhere else**: through the domain of
integration $B$ (the ε-ball) and through the sampling density $q$ inside it.
Both are already in the code — `project_delta`'s projection, `random_starts`'
uniform sampling, and the `delta_norm` that `min_norm`/`max_norm` rank by. That
last one *should* be the threat norm, because it asks "which landing is closest
to the clean image", a distance and not a surface measure. Putting L∞ into the
weight factor as well would count it twice.

The dual norm answers a different question: to first order, how far you must
move *in the threat norm* to change $m$ by $\Delta$,

$$\min\{\lVert\delta\rVert_p : \nabla m\cdot\delta = -\Delta\} = \Delta / \lVert\nabla m\rVert_q, \qquad 1/p + 1/q = 1$$

which is the MMA / DeepFool margin-to-radius conversion, not a surface measure.

The two are **not** a rescaling of each other. Since
$\lVert g\rVert_2 \le \lVert g\rVert_1 \le \sqrt{d}\,\lVert g\rVert_2$, the ratio
$\lVert g\rVert_1/(\sqrt{d}\lVert g\rVert_2)$ is precisely the gradient's
*spread*: 1 for a gradient smeared evenly over all pixels, $1/\sqrt d$ for a
1-sparse one. With $d = 3072$ the two can differ by up to ~55×, and they
genuinely **reorder** the landing positions — L1 prefers positions whose gradient
is spread across many pixels, which is what an L∞ attacker can exploit, while L2
is indifferent to spread. So `dual` is a real ablation axis (runs `w_sharp_dual`
and `w_flat_dual`), just not the principled default.

## Results

<!-- BEGIN AUTO:screen -->
Not run yet.
<!-- END AUTO -->

<!-- BEGIN AUTO:confirm -->
Not run yet.
<!-- END AUTO -->

## Reading the table

Check the last three columns **before** the accuracy columns.

- **`valid`** (`valid_position_rate`) — fraction of perturbations that actually
  reached the level set. Below ~0.3 the geometry columns describe points that
  are not on the level being studied, and the row is not evidence about its
  configuration. It is a broken run, not a bad configuration.
- **`ae`** (`ae_rate_valid`) — fraction of the *weighted* positions that are
  genuinely misclassified (m < 0). Should be ≈ 1.000. Anything lower means part
  of the training signal came from points that are not adversarial examples,
  which is the premise of the whole loss.
- **`eff_rank`** — effective rank of the pairwise-cosine Gram matrix over the
  valid landings. Near 1 means the starts collapsed onto one direction; that
  makes the `weight_mode` comparison vacuous even when `valid` looks fine.

Then read `clean−PGD` alongside `PGD-10`: a configuration that wins on robust
accuracy purely by giving up clean accuracy has not shown anything the trade-off
curve does not already offer.

FGSM is a single step and so upper-bounds PGD-10. A large FGSM−PGD gap is the
usual sign of gradient masking, and a robustness number obtained that way will
not survive AutoAttack.

## Confirmed effective configuration

*To be filled in after the `confirm` stage.* It should contain a copy-pastable
command line, the margin over the strongest baseline, and the specific numbers
the claim rests on — not just the winning tag.

```
(pending)
```
