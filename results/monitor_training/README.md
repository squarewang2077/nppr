# Monitor training — results

Plots and notes for the metrics logged during training
(`ckp/monitor_training/.../*_training_info.csv`). See
[`plot_training_info.ipynb`](./plot_training_info.ipynb) for the plotting code.

## PGD path-drift metrics: `path_drift_step`, `path_drift_endpoint`, `path_cos`

These three columns are defined in `path_drift()` at
[`scripts/monitor_training.py:299-323`](../../scripts/monitor_training.py#L299-L323).

### What they measure

When training is run with `--track_path`, every epoch runs a PGD attack on a
**fixed batch of images** and records the full **attack trajectory** (not just the
final adversarial example):

$$\Delta^e(x) = [\delta_1, \delta_2, \dots, \delta_T], \qquad \delta_t = x^{adv}_t - x$$

i.e. the perturbation after PGD step $t$. Stored as a tensor of shape
`(T, B, C, H, W)` (T = PGD steps, B = images); see `compute_pgd_path`.

The current epoch's trajectory $\Delta^e$ is then compared **step-by-step** with
the previous epoch's trajectory $\Delta^{e-1}$. So these metrics answer:
**"after this epoch's weight update, how much did the PGD attack path the model
induces change relative to the previous epoch?"** Because `random_start` defaults
to OFF, both paths share the same starting point, so the drift reflects the model
update alone.

### How they are computed

Let $d = C\cdot H\cdot W$ (per-image element count) and the normalizing
denominator $\text{denom} = \epsilon\sqrt{d}$. First the per-step, per-image L2
distance `step_l2`:

$$\text{step\_l2}[t,b] = \lVert \delta_t^e - \delta_t^{e-1}\rVert_2$$

**1. `path_drift_step`** — mean per-step drift

$$\frac{1}{T}\sum_{t=1}^{T} \operatorname{mean}_B \frac{\lVert \delta_t^e - \delta_t^{e-1}\rVert_2}{\epsilon\sqrt{d}}$$

Averaged over **all steps and all images** (`step_l2.mean() / denom`). Captures how
much the whole trajectory shifted on average.

**2. `path_drift_endpoint`** — endpoint drift

$$\operatorname{mean}_B \frac{\lVert \delta_T^e - \delta_T^{e-1}\rVert_2}{\epsilon\sqrt{d}}$$

Uses only the **final step $t=T$** (the final adversarial example),
`step_l2[-1].mean() / denom`. Captures how much the final adversarial point moved.

**3. `path_cos`** — path-direction cosine similarity

$$\operatorname{mean}_{t,B}\ \cos\!\big(\delta_t^e,\ \delta_t^{e-1}\big)$$

Cosine similarity between the flattened ($d$-dim) perturbation vectors of
$\Delta^e$ and $\Delta^{e-1}$ at each step, averaged over all $(t, B)$
(`F.cosine_similarity(..., dim=2)`). Captures whether the attack **direction** is
stable.

### Normalization and value ranges

- The first two metrics are divided by $\epsilon\sqrt{d}$, giving a **per-element
  RMS deviation measured in units of $\epsilon$**. For an L∞ attack (each pixel
  perturbation $\in[-\epsilon, \epsilon]$) this lies in **[0, 2]**, so runs with
  different $\epsilon$ or resolution are directly comparable.
- `path_cos` needs no normalization; its range is **[-1, 1]**. Close to 1 means the
  two epochs' attacks point in almost the same direction (training stabilizing); a
  drop means the attack direction is changing sharply.

### Intuition

| Metric | Larger means | Expected as training converges |
|---|---|---|
| `path_drift_step` | the whole attack path changes more each epoch | decreases → 0 |
| `path_drift_endpoint` | the final adversarial point is less stable | decreases → 0 |
| `path_cos` | the attack direction is more consistent (1 = identical) | increases → 1 |

Because `path_cos` and the two drift metrics are on different scales, the notebook
plots them together with `normalize=True` to compare their trends/shapes.
