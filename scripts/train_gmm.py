#!/usr/bin/env python3
"""
Main Script to train the GMM of NPPR. The configuration is handled via config.py.

Usage:
    python fit_gmm.py --config <config_name>
"""


import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

# Import your classes and utilities
from configs.train_gmm_cfg import get_config, list_configs, initialize_gmm_parameters, TemperatureScheduler
from src.gmm4pr import GMM4PR, build_decoder_from_flag as build_decoder
from arch import build_model, build_feat_extractor
from utils import get_dataset_with_index, check_mode_collapse


def main():
    # ============ PARSE ARGUMENTS ============
    parser = argparse.ArgumentParser(description="GMM4PR Training")
    
    # Config selection
    parser.add_argument("--config", type=str, default="mobilenet_on_cifar10",
                       help="Config name from config.py")
    parser.add_argument("--list-configs", action="store_true", default=False, # false for debug
                       help="List all available configs and exit")
    
    # Quick overrides (optional)
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--K", type=int, help="Override number of components")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--device", type=str, help="Override device")
    parser.add_argument("--clf_ckpt", type=str, help="Override classifier checkpoint path")
    parser.add_argument("--ckp_dir", type=str, help="Override checkpoint save directory")
    
    args = parser.parse_args()
    
    # List configs and exit
    if args.list_configs:
        list_configs()
        return
    
    # Load config
    cfg = get_config(args.config)
    
    # Apply command line overrides
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.K is not None:
        cfg.K = args.K
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.device is not None:
        cfg.device = args.device
    if args.clf_ckpt is not None:
        cfg.clf_ckpt = args.clf_ckpt
    if args.ckp_dir is not None:
        cfg.ckp_dir = args.ckp_dir
    
    # Print configuration
    print(cfg)
    
    # ============ SETUP ============
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Set random seed for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load dataset
    print(f"\nLoading dataset: {cfg.dataset}")
    dataset, num_classes, out_shape = get_dataset_with_index(cfg.dataset, cfg.data_root, train=True, resize=cfg.resize) # training set
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=cfg.batch_size,
        shuffle=False, 
        num_workers=cfg.num_workers, 
        pin_memory=True 
    )
    print(f"Dataset: {len(dataset)} samples, {num_classes} classes, shape={out_shape}")

    # ============ LOAD CLASSIFIER ============
    print(f"\nLoading classifier: {cfg.arch}")
    model = build_model(cfg.arch, num_classes, cfg.dataset)

    if not os.path.isfile(cfg.clf_ckpt):
        raise FileNotFoundError(f"Classifier not found: {cfg.clf_ckpt}")

    state = torch.load(cfg.clf_ckpt, map_location="cpu")
    state = state.get("state_dict", state.get("model_state", state))
    state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    # Build feat_extractor from model's loaded backbone so they share the same
    # parameter objects — no separate weight loading needed.
    feat_extractor = build_feat_extractor(cfg.arch, num_classes, cfg.dataset,
                                          backbone=model.backbone)
        
    # Check parameter sharing
    model_params = {id(p) for p in model.parameters()}
    feat_params  = {id(p) for p in feat_extractor.parameters()}
    shared = model_params & feat_params

    print(f"[check] model params: {len(model_params)}, feat_extractor params: {len(feat_params)}")
    if shared:
        print(f"[check] They share {len(shared)} parameters.")
    else:
        print("[check] No shared parameters.")

    # ============ INITIALIZE GMM ============
    print(f"\nInitializing GMM: K={cfg.K}, D={cfg.latent_dim}, cond={cfg.cond_mode}")
    gmm = GMM4PR(
        K=cfg.K,
        latent_dim=cfg.latent_dim,
        device=device,
        T_pi=cfg.T_pi_init,
        T_mu=cfg.T_mu_init,
        T_sigma=cfg.T_sigma_init,
        T_shared=cfg.T_shared_init
    )
    
    # Set label embedding
    if cfg.use_y_embedding:
        gmm.set_y_embedding(
            num_cls=num_classes,
            y_dim=cfg.y_emb_dim,
            normalize=cfg.y_emb_normalize
        )
    
    # Set regularization
    gmm.set_regularization(
        pi_entropy=cfg.reg_pi_entropy,
        mean_diversity=cfg.reg_mean_div,
    )
    
    # Infer feature dimension if needed
    feat_dim = None
    if cfg.cond_mode in ("x", "xy"):
        with torch.no_grad():
            x0, _, _ = next(iter(loader))
            feat_dim = feat_extractor(x0.to(device)).view(x0.size(0), -1).size(1)
        print(f"Feature dimension: {feat_dim}")
    
    # Set conditioning
    gmm.set_condition(
        cond_mode=cfg.cond_mode,
        cov_type=cfg.cov_type,
        cov_rank=cfg.cov_rank,
        feat_dim=feat_dim or 0,
        num_cls=num_classes,
        hidden_dim=cfg.hidden_dim
    )
    
    # Set feature extractor
    if cfg.cond_mode in ("x", "xy"):
        gmm.set_feat_extractor(feat_extractor)
    
    # Set decoder
    if cfg.use_decoder:
        decoder = build_decoder(
            cfg.decoder_backend,
            cfg.latent_dim,
            out_shape,
            device
        )
        gmm.set_up_sampler(decoder)
    
    # Set budget
    gmm.set_budget(norm=cfg.norm, eps=cfg.epsilon)
    
    # Initialize parameters
    initialize_gmm_parameters(gmm, init_mode=cfg.init_mode) # only for unconditional GMM
    
    # Temperature scheduler
    temp_scheduler = TemperatureScheduler(
        gmm,
        initial_T_pi=cfg.T_pi_init,
        final_T_pi=cfg.T_pi_final,

        initial_T_mu=cfg.T_mu_init,
        final_T_mu=cfg.T_mu_final,

        initial_T_sigma=cfg.T_sigma_init,
        final_T_sigma=cfg.T_sigma_final,

        initial_T_shared=cfg.T_shared_init,
        final_T_shared=cfg.T_shared_final,

        warmup_epochs=cfg.warmup_epochs
    )
    
    # ============ OPTIMIZER ============
    optimizer = optim.Adam(
        [p for p in gmm.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    # ============ LEARNING RATE SCHEDULER ============
    scheduler = None
    if cfg.use_lr_scheduler:
        # Cosine Annealing with Warmup
        # We'll use CosineAnnealingLR and manually handle warmup
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        # Warmup scheduler: linearly increase LR from 0 to initial LR
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,  # Start at 1% of initial LR
            end_factor=1.0,     # End at 100% of initial LR
            total_iters=cfg.lr_warmup_epochs
        )

        # Cosine annealing scheduler: decrease LR from initial to min
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(cfg.epochs - cfg.lr_warmup_epochs, 1),  # Remaining epochs after warmup
            eta_min=cfg.lr_min
        )

        # Combine warmup and cosine annealing
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[cfg.lr_warmup_epochs]
        )

        print(f"\nLearning rate scheduler enabled:")
        print(f"  Warmup epochs: {cfg.lr_warmup_epochs}")
        print(f"  Initial LR: {cfg.lr}")
        print(f"  Min LR: {cfg.lr_min}")

    # ============ TRAINING LOOP ============
    os.makedirs(cfg.ckp_dir, exist_ok=True)
    collapse_log = []
    loss_hist = {"epoch": [],
                 "loss": [],
                 "main_loss": [],
                 "reg_loss": [],
                 "pr": [],
                 "learning_rate": []
                 } # To store loss history

    gmm.train()
    print(f"\n{'='*60}")
    print(f"Starting training: {cfg.epochs} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(1, cfg.epochs + 1):
        # Update temperatures for distribution parameters
        T_pi, T_mu, T_sigma, T_shared = temp_scheduler.step(epoch)
        
        # Compute Gumbel temperature (optional annealing)
        if hasattr(cfg, 'use_gumbel_anneal') and cfg.use_gumbel_anneal:
            alpha = (epoch - 1) / max(cfg.epochs - 1, 1)
            gumbel_temp = cfg.gumbel_temp_init + alpha * (cfg.gumbel_temp_final - cfg.gumbel_temp_init)
        else:
            gumbel_temp = cfg.gumbel_temp_final  # Fixed temperature
        
        # Progress bar
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs} [norm={cfg.norm}, eps={cfg.epsilon:.3f}]")
        
        # Metrics
        epoch_loss = 0.0
        epoch_main = 0.0
        epoch_reg = 0.0
        epoch_pr = 0.0
        epoch_pr_count = 0  # Track total samples for PR calculation
        total_samples = 0
        num_processed_batches = 0 # to account for skipped batches

        acc_counter = 0
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (x, y, _) in enumerate(pbar):
            if batch_idx >= cfg.batch_index_max: # this line added for testing 
                break
            x, y = x.to(device), y.to(device)
            
            # Only use correctly classified samples
            with torch.no_grad():
                model.eval()
                pred = model(x).argmax(1)
                mask = (pred == y).tolist()
                # debug_mask = [False] * 256
                # debug_mask[0] = True
                if sum(mask) == 0:
                    continue                
            x_clean = x[mask] 
            y_clean = y[mask]
            total_samples += len(y_clean)
            num_processed_batches += 1

            # Compute loss (return details on first processed batch of checkpoint epochs)
            return_details = (num_processed_batches == 1 and epoch % cfg.check_collapse_every == 0)
            
            out = gmm.pr_loss(
                x_clean, y_clean, model,
                num_samples=cfg.num_samples,
                loss_variant=cfg.loss_variant, kappa=cfg.kappa,
                chunk_size=cfg.chunk_size,
                return_reg_details=return_details,
                gumbel_temperature=gumbel_temp  # Pass Gumbel temperature
            )
            
            loss = out["loss"] / cfg.accumulate_grad
            loss.backward()
            acc_counter += 1

            # Gradient step
            if acc_counter % cfg.accumulate_grad == 0:
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(gmm.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Accumulate metrics
            epoch_loss += out["loss"].item()
            epoch_main += out["main"].item()
            epoch_reg += out["reg"].item()
            epoch_pr += out["pr"] * len(y_clean)  # Weight by batch size
            epoch_pr_count += len(y_clean)  # Track total samples
            
            # Print regularization details
            if return_details and 'reg_details' in out:
                print(f"\n[Epoch {epoch}] Regularization details:")
                for k, v in out['reg_details'].items():
                    print(f"  {k:20s}: {v:.6f}")
                print(f"  π distribution: {out['pi_probs'].cpu().numpy()}")
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{out['loss'].item():.4e}",
                "main": f"{out['main'].item():.4e}",
                "reg": f"{out['reg'].item():.4e}",
            })
        
        # Final gradient step
        if acc_counter % cfg.accumulate_grad != 0:
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(gmm.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Epoch summary
        avg_loss = epoch_loss / max(num_processed_batches, 1)
        avg_main = epoch_main / max(num_processed_batches, 1)
        avg_reg = epoch_reg / max(num_processed_batches, 1)
        avg_pr = epoch_pr / max(epoch_pr_count, 1)  # Divide by total samples, not batches

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Record loss history
        loss_hist["epoch"].append(epoch)
        loss_hist["loss"].append(avg_loss)
        loss_hist["main_loss"].append(avg_main)
        loss_hist["reg_loss"].append(avg_reg)
        loss_hist["pr"].append(avg_pr)
        loss_hist["learning_rate"].append(current_lr)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Loss: {avg_loss:.4f} (main={avg_main:.4f}, reg={avg_reg:.4f})")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Batches: {num_processed_batches}/{len(loader)} processed")
        print(f"  Samples used: {total_samples}/{len(dataset)}")
        print(f"  Temperatures: T_pi={T_pi:.2f}, T_mu={T_mu:.2f}, T_sigma={T_sigma:.2f}, T_shared={T_shared:.2f}, T_gumbel={gumbel_temp:.2f}")

        # Step the learning rate scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Check mode collapse
        if epoch % cfg.check_collapse_every == 0:
            stats = check_mode_collapse(gmm, loader, device)
            gmm.train()
            collapse_log.append({
                'epoch': epoch,
                'max_pi': stats['max_pi'],
                'min_pi': stats['min_pi'],
                'std_pi': stats['std_pi'],
                'entropy_ratio': stats['entropy_ratio'],
                'T_pi': T_pi,
                'T_gumbel': gumbel_temp,
                'avg_loss': avg_loss
            })

    # ======= SAVE ============
    print(f"\n{'='*60}")
    print("Training complete! Saving model...")
    print(f"{'='*60}")
    
    # saving directory
    save_dir = f"{cfg.ckp_dir}/{cfg.arch}_on_{cfg.dataset}/"
    os.makedirs(save_dir, exist_ok=True)

    # save the training loss
    pd.DataFrame(loss_hist).to_csv(os.path.join(save_dir, f"loss_hist_{cfg.exp_name}.csv"), index=False)
    print(f"[save] loss history -> {save_dir}/loss_hist_{cfg.exp_name}.csv")

    # Save model with metadata (including final gumbel_temp and configuration for reference)
    save_path = os.path.join(save_dir, f"gmm_{cfg.exp_name}.pt")
    gmm.save(
        save_path, 
        extra={
            "config": cfg.to_dict(),
            "final_gumbel_temperature": gumbel_temp,  # Final value used
        }
    )
    print(f"✓ Model saved: {save_path}")
    
    # Save collapse log
    if collapse_log:
        df = pd.DataFrame(collapse_log)
        log_path = os.path.join(save_dir, f"collapse_log_{cfg.exp_name}.csv")
        df.to_csv(log_path, index=False)
        print(f"✓ Collapse log saved: {log_path}")
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()