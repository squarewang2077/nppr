#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper script for robustness evaluation using ar_and_pr scripts.
Compatible with the Config system used in the main codebase.

This script evaluates trained classifiers using:
- PGD (Projected Gradient Descent) attacks
- CW (Carlini-Wagner) attacks
- PR (Probabilistic Robustness) evaluation

Usage Examples:

    # DEBUG MODE (uses defaults, all evaluations skipped):
    python eval_robustness.py

    # Run from IDE/debugger with default settings:
    # Just click "Run" - no args needed!

    # QUICK TESTING (run specific evaluation):
    python eval_robustness.py --run_pgd        # Only PGD
    python eval_robustness.py --run_cw         # Only CW
    python eval_robustness.py --run_all        # All evaluations

    # FULL EVALUATION (specify model and config):
    python eval_robustness.py \
        --config resnet18_on_cifar10_linf_4 \
        --model_path ./model_zoo/trained_model/resnet18_cifar10.pth \
        --run_all

    # CUSTOM EVALUATION (different model, specific tests):
    python eval_robustness.py \
        --config resnet50_on_cifar100 \
        --model_path ./model_zoo/trained_model/resnet50_cifar100.pth \
        --run_pgd --run_cw

    # CUSTOM EPSILON VALUES:
    python eval_robustness.py --run_all --epsilon_values 2.0 4.0 8.0

    # CUSTOM PR DISTRIBUTIONS:
    python eval_robustness.py --run_pr --pr_distributions Uniform Normal Laplace

    # FULL CUSTOM EVALUATION:
    python eval_robustness.py \
        --config resnet18_on_cifar10_linf_4 \
        --model_path ./model_zoo/trained_model/resnet18_cifar10.pth \
        --run_all \
        --epsilon_values 4.0 8.0 16.0 \
        --pr_distributions Uniform Normal
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from configs.train_gmm_cfg import get_config
from ar_and_pr.evaluate import evaluate_PGD, evaluate_PR, evaluate_cw


class Args:
    """Simple namespace to hold evaluation arguments compatible with ar_and_pr scripts."""
    def __init__(self, config):
        # Dataset settings
        self.dataset = config.dataset.upper()
        self.data_root = config.data_root

        # Model settings
        self.model_name = config.arch

        # Determine number of classes and input size
        if config.dataset.lower() == 'cifar10':
            self.num_class = 10
            self.input_size = 32
        elif config.dataset.lower() == 'cifar100':
            self.num_class = 100
            self.input_size = 32
        elif config.dataset.lower() == 'tinyimagenet':
            self.num_class = 200
            self.input_size = 64
        else:
            raise ValueError(f"Unsupported dataset: {config.dataset}")

        # Attack settings (using config values)
        self.attack_steps = 10  # Default for evaluation
        self.attack_eps = config.epsilon * 255  # Convert to 0-255 scale
        self.attack_lr = 2.0  # Default step size

        # Other settings
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.batch_size = config.batch_size


def load_model(model_path, config, device='cuda'):
    """Load a model from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    # Build model architecture
    from torchvision.models import (
        resnet18, resnet50, wide_resnet50_2, vgg16
    )

    arch = config.arch.lower()

    # Determine number of classes
    if config.dataset.lower() == 'cifar10':
        num_classes = 10
    elif config.dataset.lower() == 'cifar100':
        num_classes = 100
    elif config.dataset.lower() == 'tinyimagenet':
        num_classes = 200
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    # Build model
    print(f"Building {arch} model for {config.dataset} ({num_classes} classes)...")
    if arch == 'resnet18':
        model = resnet18(weights=None)
    elif arch == 'resnet50':
        model = resnet50(weights=None)
    elif arch == 'wide_resnet50_2':
        model = wide_resnet50_2(weights=None)
    elif arch == 'vgg16':
        model = vgg16(weights=None)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Replace classifier head
    if arch.startswith('resnet') or arch.startswith('wide_resnet'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif arch.startswith('vgg'):
        in_features = model.classifier[-1].in_features
        new_classifier = list(model.classifier[:-1]) + [nn.Linear(in_features, num_classes)]
        model.classifier = nn.Sequential(*new_classifier)

    # Load state dict
    try:
        model.load_state_dict(checkpoint['model_state'])
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model_state directly: {e}")
        print("Attempting to load checkpoint as state_dict...")
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


def main():
    """
    Main evaluation function with debugging support.

    DEBUG MODE (default):
    - Run directly: `python eval_robustness.py`
    - All evaluations skipped by default for quick testing
    - Uses default config and model path

    QUICK EVAL MODE:
    - Run specific evaluation: `python eval_robustness.py --run_pgd`
    - Run all: `python eval_robustness.py --run_all`

    PRODUCTION MODE:
    - Provide explicit args:
      `python eval_robustness.py --config <config> --model_path <path> --run_all`
    """
    parser = argparse.ArgumentParser(description="Evaluate classifier robustness")

    # DEBUG MODE: Set required=False and provide defaults for quick debugging
    # For production use, either provide command-line args or change required=True
    parser.add_argument('--config', type=str, required=False,
                        default='resnet18_on_cifar10_linf_4',
                        help='Config name (e.g., resnet18_on_cifar10_linf_4)')
    parser.add_argument('--model_path', type=str, required=False,
                        default='./model_zoo/trained_model/resnet18_cifar10.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--log_dir', type=str, default='./ar_and_pr/logs',
                        help='Directory to save evaluation logs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    # DEBUG MODE: Skip all evaluations by default for quick testing
    # Set these to False to run actual evaluations
    parser.add_argument('--skip_pgd', action='store_true', default=True,
                        help='Skip PGD evaluation (default: True for debugging)')
    parser.add_argument('--skip_cw', action='store_true', default=True,
                        help='Skip CW evaluation (default: True for debugging)')
    parser.add_argument('--skip_pr', action='store_true', default=True,
                        help='Skip PR evaluation (default: True for debugging)')

    # Options to enable specific evaluations
    parser.add_argument('--run_pgd', action='store_true', default=False,
                        help='Enable PGD evaluation (overrides --skip_pgd)')
    parser.add_argument('--run_cw', action='store_true', default=False,
                        help='Enable CW evaluation (overrides --skip_cw)')
    parser.add_argument('--run_pr', action='store_true', default=False,
                        help='Enable PR evaluation (overrides --skip_pr)')
    parser.add_argument('--run_all', action='store_true', default=False,
                        help='Enable all evaluations (overrides all skip flags)')

    # Epsilon and distribution settings
    parser.add_argument('--epsilon_values', type=float, nargs='+',
                        default=[4.0, 8.0, 16.0],
                        help='Epsilon values in 0-255 scale for PGD/CW (default: [4.0, 8.0, 16.0])')
    parser.add_argument('--pr_distributions', type=str, nargs='+',
                        default=['Uniform', 'Normal'],
                        choices=['Uniform', 'Normal', 'Laplace'],
                        help='Distributions for PR evaluation (default: [Uniform, Normal])')

    args_cmd = parser.parse_args()

    # Handle evaluation flags: --run_* flags override --skip_* defaults
    if args_cmd.run_all:
        args_cmd.skip_pgd = False
        args_cmd.skip_cw = False
        args_cmd.skip_pr = False
    else:
        if args_cmd.run_pgd:
            args_cmd.skip_pgd = False
        if args_cmd.run_cw:
            args_cmd.skip_cw = False
        if args_cmd.run_pr:
            args_cmd.skip_pr = False

    # Load config
    print(f"\n{'='*80}")
    print(f"Loading configuration: {args_cmd.config}")
    print(f"{'='*80}")
    config = get_config(args_cmd.config)
    print(config)

    # Setup device
    device = torch.device(args_cmd.device if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Print evaluation plan
    print(f"\n{'='*80}")
    print("Evaluation Plan:")
    print(f"  Epsilon values: {args_cmd.epsilon_values} / 255")
    print(f"  PR Distributions: {args_cmd.pr_distributions}")
    print(f"  PGD Attack:  {'✓ Enabled' if not args_cmd.skip_pgd else '✗ Skipped'}")
    print(f"  CW Attack:   {'✓ Enabled' if not args_cmd.skip_cw else '✗ Skipped'}")
    print(f"  PR Eval:     {'✓ Enabled' if not args_cmd.skip_pr else '✗ Skipped'}")
    print(f"{'='*80}\n")

    # Load model
    model = load_model(args_cmd.model_path, config, device)

    # Create Args object compatible with ar_and_pr scripts
    args = Args(config)

    # Create optimizer (required by evaluation functions)
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # Setup logging
    os.makedirs(args_cmd.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"eval_{config.arch}_{config.dataset}_{timestamp}.log"
    eval_file = os.path.join(args_cmd.log_dir, log_filename)

    print(f"\nEvaluation log will be saved to: {eval_file}")

    # Define evaluation settings from command-line arguments
    # Multiple epsilon values: 4/255, 8/255, 16/255 (default)
    epsilon_list_255 = args_cmd.epsilon_values  # For PGD/CW (in 0-255 scale)
    epsilon_list_01 = [round(e/255, 4) for e in epsilon_list_255]  # For PR (in 0-1 scale)

    # Multiple distributions for PR evaluation
    distribution_list = args_cmd.pr_distributions

    # Write header to log file
    with open(eval_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write(f"Robustness Evaluation Report\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {config.arch}\n")
        f.write(f"Dataset: {config.dataset}\n")
        f.write(f"Model Path: {args_cmd.model_path}\n")
        f.write(f"Epsilon values (L-inf): {epsilon_list_255} / 255 = {[f'{e:.4f}' for e in epsilon_list_01]}\n")
        f.write(f"PR Distributions: {distribution_list}\n")
        f.write("="*100 + "\n\n")

    # Run evaluations
    evaluations_run = []

    if not args_cmd.skip_pgd:
        print("\n" + "="*80)
        print(f"Running PGD Evaluation with epsilon={epsilon_list_255}...")
        print("="*80)
        evaluate_PGD(args, model, optimizer, log_file=eval_file,
                    data_root=config.data_root, epsilon_list=epsilon_list_255)
        evaluations_run.append("PGD")

    if not args_cmd.skip_cw:
        print("\n" + "="*80)
        print(f"Running CW Evaluation with epsilon={epsilon_list_255}...")
        print("="*80)
        evaluate_cw(args, model, optimizer, log_file=eval_file,
                   data_root=config.data_root, epsilon_list=epsilon_list_255)
        evaluations_run.append("CW")

    if not args_cmd.skip_pr:
        print("\n" + "="*80)
        print(f"Running PR Evaluation with epsilon={epsilon_list_01} and distributions={distribution_list}...")
        print("(this may take a while)")
        print("="*80)
        evaluate_PR(args, model, log_file=eval_file, GE=False,
                   data_root=config.data_root, img_size=args.input_size,
                   epsilon_list=epsilon_list_01, distribution_list=distribution_list)
        evaluations_run.append("PR")

    # Summary
    print("\n" + "="*80)
    if evaluations_run:
        print(f"Evaluation complete!")
        print(f"  Ran: {', '.join(evaluations_run)}")
        print(f"  Results saved to: {eval_file}")
    else:
        print("No evaluations were run (all skipped for debugging).")
        print(f"  To run evaluations, use:")
        print(f"    --run_pgd    # Run PGD attack evaluation")
        print(f"    --run_cw     # Run CW attack evaluation")
        print(f"    --run_pr     # Run PR evaluation (slow)")
        print(f"    --run_all    # Run all evaluations")
    print("="*80)


if __name__ == "__main__":
    main()
