# Pseudocode Summary

---

## 1. Data Pipeline & Normalised Model

### 1.1 Dataset Loading (`utils/data_preprocessing.py`)

```
FUNCTION get_dataset(name, root, train, img_size):
    # Normalization stats are NOT applied here — they live inside the model
    tf = Compose([ Resize(img_size), ToTensor() ])   # outputs raw [0, 1] tensors

    IF name == "cifar10"  : ds = CIFAR10(root, train, tf);  num_classes = 10
    IF name == "cifar100" : ds = CIFAR100(root, train, tf); num_classes = 100
    IF name == "tinyimagenet": ds = ImageFolder(root/tiny-imagenet-200/split, tf); num_classes = 200

    RETURN ds, num_classes


FUNCTION get_img_size(dataset, manual_override=None):
    IF manual_override is not None: RETURN manual_override
    IF dataset == "tinyimagenet": RETURN 64
    RETURN 32   # CIFAR-10/100 native size


FUNCTION get_norm_stats(dataset):
    RETURN (mean, std) for the given dataset   # used by NormalizedModel
```

### 1.2 Normalised Model Wrapper (`model_zoo/__init__.py`)

```
CLASS NormalizedModel(backbone, mean, std):
    """
    Wraps any backbone so that:
      - Input:  raw image tensors in [0, 1]
      - First operation: (x - mean) / std  (channel-wise, as buffer on device)
      - Then: backbone forward pass

    This keeps the attack code simple (always works in [0,1] space) while
    the model sees properly normalised values.
    """
    buffers: mean (1,3,1,1), std (1,3,1,1)

    FUNCTION forward(x):
        RETURN backbone( (x - mean) / std )


FUNCTION build_model(arch, num_classes, dataset):
    backbone = MODEL_REGISTRY[arch](num_classes)
    mean, std = get_norm_stats(dataset)
    RETURN NormalizedModel(backbone, mean, std)
```

---

## 2. Classifier Training (`fit_classifiers.py`)

### 2.1 Training Loop

```
FUNCTION main():
    # --- Setup ---
    args = parse_args()
    set_seed(args.seed)
    img_size = get_img_size(args.dataset, args.img_size)

    train_set, num_classes = get_dataset(args.dataset, args.data_root, train=True,  img_size)
    test_set,  _           = get_dataset(args.dataset, args.data_root, train=False, img_size)

    model     = build_model(args.arch, num_classes, args.dataset)  # includes NormalizedModel wrapper
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    adv_config = { type, norm, epsilon, alpha, num_steps, beta }
    sigma_list = build_sigma_list(epsilon, K, sigma_dist_type)
    pr_config  = { type, norm, epsilon, beta_mix, kappa, K,
                   sigma_list, fisher_damping, tau, noise_scale, num_samples }

    # --- Training ---
    FOR ep = 1..epochs:
        IF training_type == "standard":
            loss = train_one_epoch(model, loader, optimizer, criterion)

        IF training_type in ["adv_pgd", "trades"]:
            loss = train_one_epoch_adv(model, loader, optimizer, criterion, adv_config)

        IF training_type == "pr":
            loss = train_one_epoch_pr(model, loader, optimizer, criterion, pr_config)

        acc = evaluate(model, test_loader)
        print(ep, loss, acc)

    # --- Save last checkpoint ---
    save { epoch, arch, dataset, img_size, training_type, model_state, [adv/pr]_config }
```

### 2.2 Standard Epoch (`train_one_epoch`)

```
FUNCTION train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    FOR (x, y) in loader:
        logits = model(x)          # NormalizedModel normalises internally
        loss   = criterion(logits, y)
        loss.backward(); optimizer.step(); optimizer.zero_grad()
    RETURN avg_loss
```

### 2.3 Adversarial Epoch (`train_one_epoch_adv`)

```
FUNCTION train_one_epoch_adv(model, loader, optimizer, criterion, adv_config):
    model.train()
    FOR (x, y) in loader:
        optimizer.zero_grad()

        IF adv_type == "adv_pgd":
            loss, x_adv = pgd_at_loss(model, x, y, epsilon, alpha,
                                       num_steps, criterion, norm)

        IF adv_type == "trades":
            loss, x_adv = trades_loss(model, x, y, epsilon, alpha,
                                       num_steps, beta, criterion, norm)

        loss.backward(); optimizer.step()
    RETURN avg_loss
```

### 2.4 PR Epoch (`train_one_epoch_pr`)

```
FUNCTION train_one_epoch_pr(model, loader, optimizer, criterion, pr_config):
    model.train()
    generator_kwargs = pr_config  (excluding "type" key)

    FOR (x, y) in loader:
        optimizer.zero_grad()

        x_adv = pr_generator(model, x, y, **generator_kwargs)
        # x_adv: (B, N, C, H, W)

        B, N      = x_adv.shape[:2]
        x_flat    = x_adv.view(B*N, C, H, W)   # flatten samples
        y_rep     = y.repeat_interleave(N)       # match labels

        logits = model(x_flat)                   # (B*N, num_classes)
        loss   = criterion(logits, y_rep)

        loss.backward(); optimizer.step()
    RETURN avg_loss
```

---

## 3. Adversarial Attacks (`utils/adv_attacker.py`)

### 3.1 Shared Helpers

```
FUNCTION _l2_random_init(x, epsilon):
    noise = randn_like(x)
    noise = noise / ||noise||_2  * epsilon   # project to L2 sphere
    RETURN clamp(x + noise, 0, 1)


FUNCTION _l2_step(x_adv, x, grad, alpha, epsilon):
    grad_unit = grad / ||grad||_2             # unit-normalise gradient (per sample)
    x_adv     = x_adv + alpha * grad_unit
    delta     = x_adv - x
    delta     = delta * min(1, epsilon / ||delta||_2)   # project onto L2 ball
    RETURN clamp(x + delta, 0, 1)


FUNCTION _linf_step(x_adv, x, grad, alpha, epsilon):
    x_adv = x_adv + alpha * sign(grad)
    delta = clamp(x_adv - x, -epsilon, epsilon)         # project onto L-inf ball
    RETURN clamp(x + delta, 0, 1)
```

### 3.2 PGD Attack (inner loop)

```
FUNCTION pgd_attack(model, x, y, epsilon, alpha, num_steps, norm, random_start=True):
    x_adv = x.clone()

    IF random_start:
        IF norm == "linf": x_adv = x_adv + Uniform(-ε, ε); clamp(0,1)
        IF norm == "l2":   x_adv = _l2_random_init(x_adv, epsilon)

    FOR step = 1..num_steps:
        loss = CrossEntropy(model(x_adv), y)
        grad = ∇_{x_adv} loss

        IF norm == "linf": x_adv = _linf_step(x_adv, x, grad, alpha, epsilon)
        IF norm == "l2":   x_adv = _l2_step(x_adv, x, grad, alpha, epsilon)

    RETURN x_adv
```

### 3.3 PGD-AT Loss (outer loop)

```
FUNCTION pgd_at_loss(model, x, y, epsilon, alpha, num_steps, criterion, norm):
    """Madry et al., ICLR 2018"""
    model.eval()
    x_adv = pgd_attack(model, x, y, epsilon, alpha, num_steps, norm)

    model.train()
    loss = criterion(model(x_adv), y)
    RETURN loss, x_adv
```

### 3.4 TRADES Loss (outer loop)

```
FUNCTION trades_loss(model, x, y, epsilon, alpha, num_steps, beta, criterion, norm):
    """Zhang et al., ICML 2019  —  Loss = CE(f(x), y) + β · KL(f(x) ∥ f(x_adv))"""
    model.eval()

    # Freeze clean distribution as KL target
    p_clean = softmax(model(x))

    # Random start
    IF norm == "linf": x_adv = x + Uniform(-ε, ε); clamp(0,1)
    IF norm == "l2":   x_adv = _l2_random_init(x, epsilon)

    # Inner loop: maximise KL divergence
    FOR step = 1..num_steps:
        loss_kl = KL( log_softmax(model(x_adv)) ∥ p_clean )
        grad    = ∇_{x_adv} loss_kl

        IF norm == "linf": x_adv = _linf_step(x_adv, x, grad, alpha, epsilon)
        IF norm == "l2":   x_adv = _l2_step(x_adv, x, grad, alpha, epsilon)

    model.train()

    # Outer loss
    loss_natural = criterion(model(x), y)
    loss_robust  = KL( log_softmax(model(x_adv)) ∥ softmax(model(x)) )
    RETURN loss_natural + beta * loss_robust,  x_adv
```

---

## 4. PR Perturbation Generator (`utils/pr_generator.py`)

```
Algorithm: Bayesian MoG Perturbation Generator (per minibatch)

Inputs:
  model       — classifier (NormalizedModel, expects [0,1] inputs)
  x           — clean inputs (B, C, H, W) in [0, 1]
  y           — labels (B,)
  epsilon     — perturbation budget radius
  norm        — "linf" | "l2"
  beta_mix    — interpolation weight ∈ [0,1] (CE ↔ soft-0-1)
  kappa       — margin softness > 0
  K           — number of MoG components
  sigma_list  — prior std per component [σ_1, …, σ_K]
  fisher_damping — diagonal Fisher damping > 0
  tau         — temperature > 0
  num_samples — N: number of perturbation samples per image
  noise_scale — posterior sampling noise scale (0 = deterministic)


STEP 1 — Compute beta-mixed gradient  g = ∇_x ℓ'_{β,κ}
------------------------------------------------------------
  model.eval()
  logits = model(x)

  ce     = CrossEntropy(logits, y, reduction="none")   # (B,)

  # Soft 0-1 margin surrogate: sigmoid(-(f_y - max_{j≠y} f_j) / κ)
  f_y     = logits[:, y]
  f_other = max logit over incorrect classes
  soft01  = sigmoid( -(f_y - f_other) / kappa )        # (B,)

  mixed_loss = mean( (1 - beta_mix)*ce + beta_mix*soft01 )

  g = ∇_x mixed_loss    # (B, C, H, W) — gradient wrt input


STEP 2 — Diagonal Fisher curvature  A = g² + damping
------------------------------------------------------------
  model.train()
  A      = g² + fisher_damping    # (B, C, H, W)
  g_flat = g.view(B, -1)          # (B, d)
  A_flat = A.view(B, -1)          # (B, d)


STEP 3 — Closed-form MoG posterior (prior μ₀ = 0, Σ₀ = σ_k² I)
------------------------------------------------------------
  FOR k = 1..K:
      prec0  = 1 / σ_k²

      # Posterior variance (diagonal):   Σ*_k = (prec0 + A/τ)^{-1}
      sig_k  = 1 / (prec0 + A_flat / tau)              # (B, d)

      # Posterior mean (μ₀=0 → simplifies):   μ*_k = Σ*_k · (g/τ)
      mu_k   = sig_k * (g_flat / tau)                  # (B, d)

      # Simplified log-evidence (fast approximation, no logdet term):
      logZ_k = 0.5 * sum( mu_k * (g_flat / tau) )      # (B,)

      log_w_k = log(π₀_k) + logZ_k

  π*  = softmax( [log_w_1, …, log_w_K] )               # (B, K)


STEP 4 — Vectorised sampling  (N samples per image)
------------------------------------------------------------
  k_idx   ~ Categorical(π*)      shape (B, N)  — component indices

  mu_sel  = gather(mu_all,  k_idx)   # (B, N, d)
  sig_sel = gather(sig_all, k_idx)   # (B, N, d)

  ε_noise ~ N(0, I)                  # (B, N, d)
  δ_flat  = mu_sel + noise_scale * sqrt(sig_sel) * ε_noise   # (B, N, d)

  δ = δ_flat.view(B, N, C, H, W)


STEP 5 — Project onto perturbation budget  Π_B(δ)
------------------------------------------------------------
  IF norm == "linf":
      δ = clamp(δ, -epsilon, epsilon)                   # (B, N, C, H, W)

  IF norm == "l2":
      FOR each (b, n):
          δ[b,n] = δ[b,n] * min(1, epsilon / ||δ[b,n]||_2)

  x_adv = clamp(x.unsqueeze(1) + δ, 0, 1)              # (B, N, C, H, W)

RETURN x_adv
```

### Key design notes

| Aspect | Choice | Rationale |
|---|---|---|
| Gradient loss `ℓ'` | `(1-β)*CE + β*soft01` | Differentiable surrogate of 0-1 loss |
| Curvature | Diagonal Fisher `g²` | O(d) cost, no Hessian inversion |
| Prior mean `μ₀` | 0 | Centres prior at no perturbation |
| logZ | Simplified (no logdet) | Avoids O(d) logdet; competitive in practice |
| Output shape | `(B, N, C, H, W)` | N i.i.d. samples per image for loss averaging |
| Model mode | `eval()` for gradient, `train()` for outer loss | Avoids BatchNorm noise in gradient estimate |

---

## 5. Sigma List Builder (`config_fitting.py`)

```
FUNCTION build_sigma_list(epsilon, K, mode_type, min_ratio=0.4, rho=0.5):
    """
    Build the K prior standard deviations for the MoG perturbation generator.
    All values are fractions of the epsilon budget.
    """
    IF mode_type == "linear":
        # Evenly spaced from min_ratio*ε to ε  (recommended)
        sigma_list = linspace(min_ratio * epsilon, epsilon, K)

    IF mode_type == "geometric":
        # Geometric progression ending at ε
        sigma_list = [ epsilon * rho^(K-1-k)  for k in 0..K-1 ]

    IF mode_type == "full":
        # Evenly spaced from ε/K to ε
        sigma_list = [ epsilon * (k+1) / K  for k in 0..K-1 ]

    RETURN sigma_list   # length K
```

---

## 6. GMM Training (`fit_gmm.py`)

```
FUNCTION train_gmm(config):
    # --- Setup ---
    cfg = load_config(config_name)
    dataset = load_training_dataset(cfg.dataset)
    loader = DataLoader(dataset, batch_size=cfg.batch_size)

    # --- Load frozen classifier ---
    model, feat_extractor = build_model(cfg.arch)
    model.load_state_dict(checkpoint)
    freeze(model); freeze(feat_extractor)

    # --- Initialize GMM ---
    gmm = GMM4PR(K, latent_dim, temperatures)
    IF use_y_embedding:
        gmm.set_y_embedding(num_cls, y_emb_dim)
    gmm.set_regularization(pi_entropy, mean_diversity)
    gmm.set_condition(cond_mode, cov_type, feat_dim, hidden_dim)
    IF use_decoder:
        gmm.set_up_sampler(decoder)
    gmm.set_budget(norm, epsilon)
    initialize_gmm_parameters(gmm)

    temp_scheduler = TemperatureScheduler(gmm, T_init, T_final, warmup_epochs)
    optimizer = Adam(gmm.parameters(), lr)
    lr_scheduler = CosineAnnealingWithWarmup(optimizer)

    # --- Training loop ---
    FOR epoch = 1 TO num_epochs:
        T_pi, T_mu, T_sigma, T_shared = temp_scheduler.step(epoch)
        gumbel_temp = anneal(gumbel_temp_init -> gumbel_temp_final, epoch)

        FOR each batch (x, y) in loader:
            pred = model(x).argmax(1)
            mask = (pred == y)
            IF no correct samples: CONTINUE
            x_clean, y_clean = x[mask], y[mask]

            out = gmm.pr_loss(x_clean, y_clean, model, num_samples, gumbel_temp)
            #   1. forward(x, y) -> build GMM distribution
            #   2. rsample via Gumbel-Softmax -> latent eps [S, B, D]
            #   3. decode eps -> image-space u via decoder
            #   4. project u to budget: delta = eps * tanh(u)  [linf]
            #   5. classify: logits = model(x + delta)
            #   6. loss = CW_margin_loss(logits, y) + regularization

            loss = out["loss"] / accumulate_grad
            loss.backward()

            IF accumulation_step:
                clip_grad_norm(gmm.parameters())
                optimizer.step(); optimizer.zero_grad()

        lr_scheduler.step()
        check_mode_collapse(gmm) periodically

    save gmm checkpoint, loss history, collapse log
```

---

## 7. PR Inference / Estimation (`eva_ar_pr.py`)

```
FUNCTION estimate_pr_baseline(model, test_loader, distribution, num_samples, epsilon, norm_type):
    """
    PR = E_{(x,y) in CleanCorrect} E_{delta ~ Noise} [ 1{ f(x + delta) = y } ]
    """
    total_used = 0
    pr_sum = 0

    FOR each batch (x, y) in test_loader:
        pred = model(x).argmax(1)
        mask = (pred == y)
        IF no correct samples: CONTINUE
        x_sel, y_sel = x[mask], y[mask]
        n = |x_sel|

        per_image_success = zeros(n)
        FOR each chunk of S samples out of num_samples:
            IF distribution == 'gaussian':  noise = randn(S, n, C, H, W)
            IF distribution == 'uniform':   noise = rand(S, n, C, H, W) * 2 - 1

            IF norm_type == 'linf': perturbations = clamp(noise, -epsilon, epsilon)
            IF norm_type == 'l2':   perturbations = noise / max(||noise||_2 / epsilon, 1)

            x_perturbed = x_sel.repeat(S) + perturbations
            pred = model(x_perturbed).argmax(1)

            correct = (pred == y_sel.repeat(S))
            per_image_success += correct.reshape(S, n).sum(dim=0)

        per_image_pr = per_image_success / num_samples
        pr_sum   += per_image_pr.sum()
        total_used += n

    RETURN pr_sum / total_used
```

---

## 8. Module Relationships

```
fit_classifiers.py
    │
    ├── utils/data_preprocessing.py   get_dataset()      → raw [0,1] tensors
    │                                 get_img_size()
    │                                 get_norm_stats()   → used by model_zoo only
    │
    ├── model_zoo/__init__.py          build_model()      → NormalizedModel(backbone)
    │                                 NormalizedModel    → normalises inside forward()
    │
    ├── utils/adv_attacker.py         pgd_at_loss()      → PGD-AT (linf / l2)
    │                                 trades_loss()      → TRADES  (linf / l2)
    │
    ├── utils/pr_generator.py         pr_generator()     → MoG Bayesian generator
    │
    └── config_fitting.py             build_sigma_list() → prior σ schedule for MoG
```
