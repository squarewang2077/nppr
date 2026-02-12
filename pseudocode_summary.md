# Pseudocode Summary

## 1. GMM Training (`fit_gmm.py`)

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
    gmm.set_condition(cond_mode, cov_type, feat_dim, hidden_dim)  # builds trunk + heads
    IF use_decoder:
        gmm.set_up_sampler(decoder)
    gmm.set_budget(norm, epsilon)
    initialize_gmm_parameters(gmm)

    temp_scheduler = TemperatureScheduler(gmm, T_init, T_final, warmup_epochs)
    optimizer = Adam(gmm.parameters(), lr)
    lr_scheduler = CosineAnnealingWithWarmup(optimizer)  # optional

    # --- Training loop ---
    FOR epoch = 1 TO num_epochs:
        T_pi, T_mu, T_sigma, T_shared = temp_scheduler.step(epoch)
        gumbel_temp = anneal(gumbel_temp_init -> gumbel_temp_final, epoch)

        FOR each batch (x, y) in loader:
            # Filter to correctly classified samples only
            pred = model(x).argmax(1)
            mask = (pred == y)
            IF no correct samples: CONTINUE
            x_clean, y_clean = x[mask], y[mask]

            # Compute PR loss
            out = gmm.pr_loss(x_clean, y_clean, model, num_samples, gumbel_temp)
            #   internally:
            #     1. forward(x, y) -> build GMM distribution (condition on x/y/xy/none)
            #     2. rsample via Gumbel-Softmax -> latent eps [S, B, D]
            #     3. decode eps -> image-space u via decoder
            #     4. project u to budget: delta = eps * tanh(u)   [linf]
            #     5. classify perturbed images: logits = model(x + delta)
            #     6. loss = CW_margin_loss(logits, y) + regularization

            loss = out["loss"] / accumulate_grad
            loss.backward()

            IF accumulation_step:
                clip_grad_norm(gmm.parameters())
                optimizer.step(); optimizer.zero_grad()

        lr_scheduler.step()
        check_mode_collapse(gmm) periodically

    # --- Save ---
    save gmm checkpoint, loss history, collapse log
```

## 2. PR Inference / Estimation (`eva_ar_pr.py`)

The baseline PR estimation uses Monte Carlo sampling with simple noise distributions (no trained GMM).

```
FUNCTION estimate_pr_baseline(model, test_loader, distribution, num_samples, epsilon, norm_type):
    """
    PR = E_{(x,y) in CleanCorrect} E_{delta ~ Noise} [ 1{ f(x + delta) = y } ]
    """
    total_used = 0
    pr_sum = 0

    FOR each batch (x, y) in test_loader:
        # Step 1: Filter to correctly classified samples
        pred = model(x).argmax(1)
        mask = (pred == y)
        IF no correct samples: CONTINUE
        x_sel, y_sel = x[mask], y[mask]
        n = |x_sel|

        # Step 2: Sample perturbations (in chunks to avoid OOM)
        per_image_success = zeros(n)
        FOR each chunk of S samples out of num_samples:
            IF distribution == 'gaussian':
                noise = randn(S, n, C, H, W)
            ELIF distribution == 'uniform':
                noise = rand(S, n, C, H, W) * 2 - 1

            # Step 3: Project noise to perturbation budget
            IF norm_type == 'linf':
                perturbations = clamp(noise, -epsilon, epsilon)
            ELIF norm_type == 'l2':
                perturbations = noise / max(||noise||_2 / epsilon, 1)

            # Step 4: Create perturbed images and classify
            x_perturbed = x_sel.repeat(S) + perturbations
            pred = model(x_perturbed).argmax(1)

            # Step 5: Accumulate per-image success count
            correct = (pred == y_sel.repeat(S))  # [S * n]
            per_image_success += correct.reshape(S, n).sum(dim=0)

        # Step 6: Compute per-image PR
        per_image_pr = per_image_success / num_samples
        pr_sum += per_image_pr.sum()
        total_used += n

    # Step 7: Average PR across all clean-correct samples
    PR = pr_sum / total_used
    RETURN PR
```

### Key difference between training and inference

| Aspect | Training (`fit_gmm.py`) | Inference (`eva_ar_pr.py`) |
|--------|------------------------|---------------------------|
| Perturbation source | Learned GMM (Gumbel-Softmax sampling) | Simple noise (Gaussian or Uniform) |
| Sampling space | Latent D-dim -> decoded to image space | Directly in image space |
| Projection | `eps * tanh(u)` (differentiable) | `clamp` or L2 normalization |
| Gradient flow | Yes (trains GMM parameters) | No (`@torch.no_grad`) |
| Goal | Learn a distribution that minimizes PR (finds worst-case perturbations) | Estimate PR under a fixed noise distribution |
