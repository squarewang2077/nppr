#!/usr/bin/env python3
"""
Main Script to train the Mixture Model (mixture4pr).

Usage:
    python train_mixture.py [options]

Example:
    python train_mixture.py --arch resnet18 --dataset cifar10 --K 7 --norm linf --epsilon 0.0314
"""


import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch import optim
from tqdm import tqdm

# Import your classes and utilities
from src.mixture4pr import (
    MixedNoise4PR,
    build_decoder_from_flag as build_decoder,
    GaussianComponent,
    LaplaceComponent,
    UniformComponent,
    SaltAndPepperComponent,
    reg_pi_entropy,
    reg_mean_diversity
)
from arch import build_model, build_feat_extractor
from utils import get_dataset, get_img_size, check_mode_collapse


def build_components(args, device):
    """
    Build noise components based on parsed args.

    Args:
        args: argparse.Namespace
        device: torch.device

    Returns:
        list of NoiseComponent objects or None (for default)
    """
    if args.component_types is None:
        # Default: all Gaussian components
        return None

    components = []
    for comp_type, num_comps in args.component_types:
        if comp_type == "gaussian":
            components.append(GaussianComponent(
                K=num_comps,
                latent_dim=args.latent_dim,
                device=device,
                cov_type=args.cov_type,
                cov_rank=args.cov_rank,
                logstd_bounds=args.logstd_bounds
            ))
        elif comp_type == "laplace":
            components.append(LaplaceComponent(
                K=num_comps,
                latent_dim=args.latent_dim,
                device=device,
                logscale_bounds=args.logstd_bounds  # Reuse logstd_bounds for laplace
            ))
        elif comp_type == "uniform":
            components.append(UniformComponent(
                K=num_comps,
                latent_dim=args.latent_dim,
                device=device,
                log_half_width_bounds=args.logstd_bounds  # Reuse logstd_bounds
            ))
        elif comp_type == "salt_and_pepper":
            components.append(SaltAndPepperComponent(
                K=num_comps,
                latent_dim=args.latent_dim,
                device=device,
                log_amplitude_bounds=args.logstd_bounds  # Reuse logstd_bounds
            ))
        else:
            raise ValueError(f"Unknown component type: {comp_type}")

    return components


def generate_exp_name(args):
    """
    Auto-generate experiment name from parsed args.

    Format: K{K}_cond({mode})_comp({components})_decoder({backend}_{dim})_{norm}({eps})_reg({regs})

    Args:
        args: argparse.Namespace

    Returns:
        str: Generated experiment name
    """
    # K value
    name_parts = [f"K{args.K}"]

    # Conditioning mode
    cond_str = args.cond_mode if args.cond_mode else "none"
    name_parts.append(f"cond({cond_str})")

    # Component types
    if args.component_types is None:
        comp_str = f"gaussian_{args.cov_type}"
    else:
        comp_str = "_".join([f"{t}({n})" for t, n in args.component_types])
    name_parts.append(f"comp({comp_str})")

    # Decoder
    if args.use_decoder:
        decoder_str = f"{args.decoder_backend}_{args.latent_dim}"
    else:
        decoder_str = f"nodec_{args.latent_dim}"
    name_parts.append(f"decoder({decoder_str})")

    # Perturbation budget
    eps_255 = int(args.epsilon * 255)
    name_parts.append(f"{args.norm}({eps_255})")

    # Regularization
    reg_parts = []
    if args.reg_pi_entropy > 0:
        reg_parts.append(f"pi{args.reg_pi_entropy}")
    if args.reg_mean_div > 0:
        reg_parts.append(f"div{args.reg_mean_div}")
    reg_str = "_".join(reg_parts) if reg_parts else "none"
    name_parts.append(f"reg({reg_str})")

    return "_".join(name_parts)


def main():
    # ============ PARSE ARGUMENTS ============
    parser = argparse.ArgumentParser(description="Mixture Model Training")

    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda, cpu)")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # Dataset
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name (cifar10, cifar100, tinyimagenet)")
    parser.add_argument("--data_root", type=str, default="./dataset", help="Dataset root directory")
    parser.add_argument("--resize", action="store_true", default=False, help="Resize images to 224")

    # Model Architecture
    parser.add_argument("--arch", type=str, default="resnet18", help="Model architecture")
    parser.add_argument("--clf_ckpt", type=str, default="./tests/standard_training/resnet18_cifar10_standard.pth",
                        help="Classifier checkpoint path")

    # Feature Extractor (optional — defaults to sharing the classifier backbone)
    parser.add_argument("--feat_arch", type=str, default=None,
                        help="Feature extractor architecture. "
                             "If omitted, the classifier backbone is reused (shared weights).")
    parser.add_argument("--feat_ckpt", type=str, default=None,
                        help="Feature extractor checkpoint. "
                             "Only used when --feat_arch is set. "
                             "If omitted with --feat_arch, the extractor starts from random weights.")

    # Mixture Model settings
    parser.add_argument("--K", type=int, default=7, help="Number of mixture components")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--logstd_bounds", type=float, nargs=2, default=[-3.0, 1.0],
                        metavar=("LB", "UB"), help="Log-std bounds (lower upper)")
    # Component types: None = all Gaussian. String format: 'gaussian:1,laplace:1,uniform:1'
    parser.add_argument("--component_types", type=str, default="gaussian:1,laplace:1,uniform:1",
                        help="Component types, e.g. 'gaussian:4,laplace:3' or 'none' for all-Gaussian")

    # Gaussian-specific settings
    parser.add_argument("--cov_type", type=str, default="diag", help="Covariance type (diag, lowrank, full)")
    parser.add_argument("--cov_rank", type=int, default=8, help="Rank for lowrank covariance")

    # Conditioning
    parser.add_argument("--cond_mode", type=str, default="xy", help="Conditioning mode (x, y, xy, none)")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")

    # Label Embedding
    parser.add_argument("--use_y_embedding", action="store_true", default=True)
    parser.add_argument("--no_y_embedding", dest="use_y_embedding", action="store_false")
    parser.add_argument("--y_emb_dim", type=int, default=128)
    parser.add_argument("--y_emb_normalize", action="store_true", default=True)
    parser.add_argument("--no_y_emb_normalize", dest="y_emb_normalize", action="store_false")

    # Decoder
    parser.add_argument("--use_decoder", action="store_true", default=True)
    parser.add_argument("--no_decoder", dest="use_decoder", action="store_false")
    parser.add_argument("--decoder_backend", type=str, default="bicubic_trainable")

    # Perturbation Budget
    parser.add_argument("--norm", type=str, default="linf", help="Norm type (linf, l2)")
    parser.add_argument("--epsilon", type=float, default=16/255, help="Perturbation budget (e.g., 0.0314 for 8/255)")

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--batch_index_max", type=float, default=float("inf"), # e.g., float("inf") for all
                        help="Max batch index per epoch (default: all batches)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=5.0, help="Gradient clipping (0 to disable)")
    parser.add_argument("--accumulate_grad", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num_samples", type=int, default=32, help="Number of MC samples per image")

    # Learning Rate Scheduler
    parser.add_argument("--use_lr_scheduler", action="store_true", default=False)
    parser.add_argument("--lr_warmup_epochs", type=int, default=5)
    parser.add_argument("--lr_min", type=float, default=2e-6)

    # Loss
    parser.add_argument("--loss_variant", type=str, default="cw", help="Loss variant (cw, ce)")
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--chunk_size", type=int, default=32)

    # Regularization
    parser.add_argument("--reg_pi_entropy", type=float, default=0.0)
    parser.add_argument("--reg_mean_div", type=float, default=0.0)

    # Gumbel-Softmax Temperature
    parser.add_argument("--use_gumbel_anneal", action="store_true", default=True)
    parser.add_argument("--no_gumbel_anneal", dest="use_gumbel_anneal", action="store_false")
    parser.add_argument("--gumbel_temp_init", type=float, default=1.0)
    parser.add_argument("--gumbel_temp_final", type=float, default=0.1)

    # Monitoring & Logging
    parser.add_argument("--check_collapse_every", type=int, default=2)
    parser.add_argument("--ckp_dir", type=str, default="./results/gmm_expressivity",
                        help="Checkpoint save directory")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name (auto-generated if not set)")

    args = parser.parse_args()

    # ---- Post-parse fixups ----

    # Convert logstd_bounds list → tuple
    args.logstd_bounds = tuple(args.logstd_bounds)

    # Parse component_types string → list of (type, count) tuples, or None
    if args.component_types is None or args.component_types.lower() == "none":
        args.component_types = None
    else:
        parsed = []
        for comp_spec in args.component_types.split(','):
            comp_type, num_comps = comp_spec.split(':')
            parsed.append((comp_type.strip(), int(num_comps)))
        args.component_types = parsed

    # Convert cond_mode "none" string → None
    if isinstance(args.cond_mode, str) and args.cond_mode.lower() == "none":
        args.cond_mode = None

    # Auto-generate exp_name if not provided
    if args.exp_name is None:
        args.exp_name = generate_exp_name(args)

    # Print configuration
    print("Configuration:")
    print("=" * 60)
    for key, value in vars(args).items():
        print(f"  {key:25s}: {value}")
    print("=" * 60)

    # ============ SETUP ============
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    img_size = get_img_size(args.dataset, manual_override=224 if args.resize else None)
    dataset, num_classes = get_dataset(
        args.dataset, args.data_root, train=True, img_size=img_size, augment=False
    )
    # Determine output shape
    channels = 1 if args.dataset.lower() == 'mnist' else 3
    out_shape = (channels, img_size, img_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Dataset: {len(dataset)} samples, {num_classes} classes, shape={out_shape}")

    # ============ LOAD CLASSIFIER ============
    print(f"\nLoading classifier: {args.arch}")
    model = build_model(args.arch, num_classes, args.dataset)

    if not os.path.isfile(args.clf_ckpt):
        raise FileNotFoundError(f"Classifier not found: {args.clf_ckpt}")

    state = torch.load(args.clf_ckpt, map_location="cpu")
    state = state.get("state_dict", state.get("model_state", state))
    state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    # ============ BUILD FEATURE EXTRACTOR ============
    if args.feat_arch is None:
        # Default: reuse the classifier's backbone (shared parameters)
        feat_extractor = build_feat_extractor(
            args.arch, num_classes, args.dataset, backbone=model.backbone
        )
        print(f"[feat_extractor] Sharing backbone from classifier ({args.arch})")
    else:
        # Independent feature extractor with its own architecture and optional checkpoint
        feat_model = build_model(args.feat_arch, num_classes, args.dataset)

        if args.feat_ckpt is not None:
            if not os.path.isfile(args.feat_ckpt):
                raise FileNotFoundError(f"Feature extractor checkpoint not found: {args.feat_ckpt}")
            feat_state = torch.load(args.feat_ckpt, map_location="cpu")
            feat_state = feat_state.get("state_dict", feat_state.get("model_state", feat_state))
            feat_state = {k.replace("module.", ""): v for k, v in feat_state.items()}
            feat_model.load_state_dict(feat_state, strict=False)
            print(f"[feat_extractor] Loaded {args.feat_arch} from {args.feat_ckpt}")
        else:
            print(f"[feat_extractor] Using randomly initialised {args.feat_arch} (no checkpoint given)")

        feat_model = feat_model.to(device).eval()
        for p in feat_model.parameters():
            p.requires_grad = False

        feat_extractor = build_feat_extractor(
            args.feat_arch, num_classes, args.dataset, backbone=feat_model.backbone
        )

    # Check parameter sharing between classifier and feature extractor
    model_params = {id(p) for p in model.parameters()}
    feat_params  = {id(p) for p in feat_extractor.parameters()}
    shared = model_params & feat_params

    print(f"[check] model params: {len(model_params)}, feat_extractor params: {len(feat_params)}")
    if shared:
        print(f"[check] They share {len(shared)} parameters (classifier backbone reused).")
    else:
        print("[check] No shared parameters (independent feature extractor).")

    # ============ INITIALIZE MIXTURE MODEL ============
    print(f"\nInitializing Mixture Model: K={args.K}, D={args.latent_dim}, cond={args.cond_mode}")

    mixture = MixedNoise4PR(
        K=args.K,
        latent_dim=args.latent_dim,
        device=device,
        logstd_bounds=args.logstd_bounds
    )

    # Set label embedding
    if args.use_y_embedding:
        mixture.set_y_embedding(
            num_cls=num_classes,
            y_dim=args.y_emb_dim,
            normalize=args.y_emb_normalize
        )

    # Infer feature dimension if needed
    feat_dim = None
    if args.cond_mode in ("x", "xy"):
        with torch.no_grad():
            x0, _ = next(iter(loader))
            feat_dim = feat_extractor(x0.to(device)).view(x0.size(0), -1).size(1)
        print(f"Feature dimension: {feat_dim}")

    # Build components
    components = build_components(args, device)
    if components is not None:
        print(f"Using custom components: {[type(c).__name__ for c in components]}")

    # Set conditioning
    mixture.set_condition(
        cond_mode=args.cond_mode,
        feat_dim=feat_dim or 0,
        num_cls=num_classes,
        hidden_dim=args.hidden_dim,
        components=components
    )

    # Set feature extractor
    if args.cond_mode in ("x", "xy"):
        mixture.set_feat_extractor(feat_extractor)

    # Set decoder
    if args.use_decoder:
        decoder = build_decoder(
            args.decoder_backend,
            args.latent_dim,
            out_shape,
            device
        )
        mixture.set_up_sampler(decoder)

    # Set budget
    mixture.set_budget(norm=args.norm, eps=args.epsilon)

    # Register regularizers
    if args.reg_pi_entropy > 0:
        mixture.register_regularizer(reg_pi_entropy, coeff=args.reg_pi_entropy)
        print(f"[reg] Registered pi_entropy with coeff={args.reg_pi_entropy}")

    if args.reg_mean_div > 0:
        mixture.register_regularizer(reg_mean_diversity, coeff=args.reg_mean_div)
        print(f"[reg] Registered mean_diversity with coeff={args.reg_mean_div}")

    # ============ OPTIMIZER ============
    optimizer = optim.Adam(
        [p for p in mixture.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # ============ LEARNING RATE SCHEDULER ============
    scheduler = None
    if args.use_lr_scheduler:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        # Warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=args.lr_warmup_epochs
        )

        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs - args.lr_warmup_epochs, 1),
            eta_min=args.lr_min
        )

        # Combine warmup and cosine annealing
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.lr_warmup_epochs]
        )

        print(f"\nLearning rate scheduler enabled:")
        print(f"  Warmup epochs: {args.lr_warmup_epochs}")
        print(f"  Initial LR: {args.lr}")
        print(f"  Min LR: {args.lr_min}")

    # ============ TRAINING LOOP ============
    os.makedirs(args.ckp_dir, exist_ok=True)
    collapse_log = []
    loss_hist = {
        "epoch": [],
        "loss": [],
        "main_loss": [],
        "reg_loss": [],
        "pr": [],
        "learning_rate": []
    }

    mixture.train()
    model.eval()
    if mixture.feat_extractor is not None:
        mixture.feat_extractor.eval()  # Ensure feature extractor is in eval mode if used

    print(f"\n{'='*60}")
    print(f"Starting training: {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        # Compute Gumbel temperature (optional annealing)
        if args.use_gumbel_anneal:
            alpha = (epoch - 1) / max(args.epochs - 1, 1)
            gumbel_temp = args.gumbel_temp_init + alpha * (args.gumbel_temp_final - args.gumbel_temp_init)
        else:
            gumbel_temp = args.gumbel_temp_final  # Fixed temperature

        # Progress bar
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs} [norm={args.norm}, eps={args.epsilon:.3f}]")

        # Metrics
        epoch_loss = 0.0
        epoch_main = 0.0
        epoch_reg = 0.0
        epoch_pr = 0.0
        epoch_pr_count = 0
        total_samples = 0
        num_processed_batches = 0

        acc_counter = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x, y) in enumerate(pbar):
            if batch_idx >= args.batch_index_max:
                break
            x, y = x.to(device), y.to(device)

            # Only use correctly classified samples
            with torch.no_grad():
                pred = model(x).argmax(1)
                mask = (pred == y).tolist()
                if sum(mask) == 0:
                    continue
            x_clean = x[mask]
            y_clean = y[mask]
            total_samples += len(y_clean)
            num_processed_batches += 1

            # Compute loss
            return_details = (num_processed_batches == 1 and epoch % args.check_collapse_every == 0)

            out = mixture.pr_loss(
                x_clean, y_clean, model,
                num_samples=args.num_samples,
                loss_variant=args.loss_variant,
                kappa=args.kappa,
                chunk_size=args.chunk_size,
                return_reg_details=return_details,
                gumbel_temperature=gumbel_temp
            )

            loss = out["loss"] / args.accumulate_grad
            loss.backward()
            acc_counter += 1

            # Gradient step
            if acc_counter % args.accumulate_grad == 0:
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(mixture.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Accumulate metrics
            epoch_loss += out["loss"].item()
            epoch_main += out["main"].item()
            epoch_reg += out["reg"].item()
            epoch_pr += out["pr"] * len(y_clean)
            epoch_pr_count += len(y_clean)

            # Print regularization details
            if return_details and 'reg_details' in out:
                print(f"\n[Epoch {epoch}] Regularization: {out['reg_details']:.6f}")
                if 'pi_probs' in out:
                    print(f"  pi distribution: {out['pi_probs'].cpu().numpy()}")

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{out['loss'].item():.4e}",
                "main": f"{out['main'].item():.4e}",
                "reg": f"{out['reg'].item():.4e}",
            })

        # Final gradient step
        if acc_counter % args.accumulate_grad != 0:
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(mixture.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Epoch summary
        avg_loss = epoch_loss / max(num_processed_batches, 1)
        avg_main = epoch_main / max(num_processed_batches, 1)
        avg_reg = epoch_reg / max(num_processed_batches, 1)
        avg_pr = epoch_pr / max(epoch_pr_count, 1)

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
        print(f"  Gumbel Temperature: {gumbel_temp:.2f}")

        # Step the learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        # Check mode collapse
        if epoch % args.check_collapse_every == 0:
            stats = check_mode_collapse(mixture, loader, device)
            # check_mode_collapse restores mixture.train(), which also puts
            # feat_extractor back into train mode — re-freeze it to eval.
            if mixture.feat_extractor is not None:
                mixture.feat_extractor.eval()
            collapse_log.append({
                'epoch': epoch,
                'max_pi': stats['max_pi'],
                'min_pi': stats['min_pi'],
                'std_pi': stats['std_pi'],
                'entropy_ratio': stats['entropy_ratio'],
                'T_gumbel': gumbel_temp,
                'avg_loss': avg_loss
            })

    # ======= SAVE ============
    print(f"\n{'='*60}")
    print("Training complete! Saving model...")
    print(f"{'='*60}")

    # saving directory
    save_dir = f"{args.ckp_dir}/{args.arch}_on_{args.dataset}/"
    os.makedirs(save_dir, exist_ok=True)

    # save the training loss
    pd.DataFrame(loss_hist).to_csv(
        os.path.join(save_dir, f"loss_hist_{args.exp_name}.csv"), index=False
    )
    print(f"[save] loss history -> {save_dir}/loss_hist_{args.exp_name}.csv")

    # Save model with metadata
    save_path = os.path.join(save_dir, f"mixture_{args.exp_name}.pt")
    mixture.save(
        save_path,
        extra={
            "config": vars(args),
            "final_gumbel_temperature": gumbel_temp,
        }
    )
    print(f"✓ Model saved: {save_path}")

    # Save collapse log
    if collapse_log:
        df = pd.DataFrame(collapse_log)
        log_path = os.path.join(save_dir, f"collapse_log_{args.exp_name}.csv")
        df.to_csv(log_path, index=False)
        print(f"✓ Collapse log saved: {log_path}")

    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
