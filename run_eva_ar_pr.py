#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Runner script for eva_ar_pr.py robustness evaluation.
Runs the evaluation and displays wall time summary in a formatted table.

Usage:
    python run_eva_ar_pr.py \
        --dataset cifar10 \
        --arch resnet18 \
        --clf_ckpt ./model_zoo/trained_model/resnet18_cifar10.pth \
        --epsilon 0.062

Or with all options:
    python run_eva_ar_pr.py \
        --dataset cifar10 \
        --arch resnet18 \
        --clf_ckpt ./model_zoo/trained_model/resnet18_cifar10.pth \
        --epsilon 0.062 \
        --norm_type linf \
        --num_samples 100 \
        --attack_steps 20 \
        --step_size 0.00784 \
        --max_batches 10 \
        --batch_size 128
"""

import subprocess
import sys
import argparse
import time
import re
from datetime import datetime


def parse_wall_times(output):
    """Parse wall times from the evaluation output."""
    wall_times = {}

    # Parse individual wall times from the output
    # Look for patterns like "Wall time: X.XXs"
    lines = output.split('\n')

    current_section = None
    for line in lines:
        # Detect section headers
        if 'Baseline Noise - Uniform' in line:
            current_section = 'uniform'
        elif 'Baseline Noise - Gaussian' in line:
            current_section = 'gaussian'
        elif 'PGD Attack' in line and 'Evaluating' in line:
            current_section = 'pgd'
        elif 'CW Attack' in line and 'Evaluating' in line:
            current_section = 'cw'

        # Parse wall time
        if 'Wall time:' in line or 'Wall Time:' in line:
            match = re.search(r'Wall [Tt]ime:\s*([\d.]+)s', line)
            if match and current_section:
                wall_times[current_section] = float(match.group(1))

    # Also try to parse from summary section
    summary_patterns = [
        (r'Baseline Uniform PR:.*Wall Time:\s*([\d.]+)s', 'uniform'),
        (r'Baseline Gaussian PR:.*Wall Time:\s*([\d.]+)s', 'gaussian'),
        (r'PGD Robust Accuracy:.*Wall Time:\s*([\d.]+)s', 'pgd'),
        (r'CW Robust Accuracy:.*Wall Time:\s*([\d.]+)s', 'cw'),
        (r'Total Wall Time:\s*([\d.]+)s', 'total'),
    ]

    for pattern, key in summary_patterns:
        match = re.search(pattern, output)
        if match:
            wall_times[key] = float(match.group(1))

    return wall_times


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def print_wall_time_summary(wall_times, total_script_time):
    """Print a formatted wall time summary table."""
    print("\n")
    print("=" * 70)
    print("                    WALL TIME SUMMARY")
    print("=" * 70)
    print(f"{'Evaluation Method':<30} {'Wall Time':>15} {'Percentage':>15}")
    print("-" * 70)

    total = wall_times.get('total', sum(v for k, v in wall_times.items() if k != 'total'))

    methods = [
        ('Baseline (Uniform)', 'uniform'),
        ('Baseline (Gaussian)', 'gaussian'),
        ('PGD Attack', 'pgd'),
        ('CW Attack', 'cw'),
    ]

    for name, key in methods:
        if key in wall_times:
            wtime = wall_times[key]
            pct = (wtime / total * 100) if total > 0 else 0
            print(f"{name:<30} {format_time(wtime):>15} {pct:>14.1f}%")

    print("-" * 70)
    print(f"{'Total Evaluation Time':<30} {format_time(total):>15} {'100.0%':>15}")
    print(f"{'Total Script Time':<30} {format_time(total_script_time):>15}")
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Runner script for eva_ar_pr.py with wall time display',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cifar10', 'cifar100', 'tinyimagenet'],
                       help='Dataset name')
    parser.add_argument('--arch', type=str, required=True,
                       choices=['resnet18', 'resnet50', 'wide_resnet50_2', 'vgg16',
                               'densenet121', 'mobilenet_v3_large', 'efficientnet_b0', 'vit_b_16'],
                       help='Model architecture')
    parser.add_argument('--clf_ckpt', type=str, required=True,
                       help='Path to classifier checkpoint')
    parser.add_argument('--epsilon', type=float, required=True,
                       help='Perturbation budget (in 0-1 scale)')

    # Optional arguments
    parser.add_argument('--data_root', type=str, default='./dataset',
                       help='Root directory for datasets')
    parser.add_argument('--norm_type', type=str, default='linf',
                       choices=['linf', 'l2'],
                       help='Norm type for perturbation')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of Monte Carlo samples')
    parser.add_argument('--attack_steps', type=int, default=20,
                       help='Number of attack steps for PGD/CW')
    parser.add_argument('--step_size', type=float, default=0.00784,
                       help='Step size for attacks')
    parser.add_argument('--max_batches', type=int, default=10,
                       help='Maximum number of batches to evaluate')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for evaluation')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory to save logs')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Build command
    cmd = [
        sys.executable, 'eva_ar_pr.py',
        '--dataset', args.dataset,
        '--arch', args.arch,
        '--clf_ckpt', args.clf_ckpt,
        '--epsilon', str(args.epsilon),
        '--data_root', args.data_root,
        '--norm_type', args.norm_type,
        '--num_samples', str(args.num_samples),
        '--attack_steps', str(args.attack_steps),
        '--step_size', str(args.step_size),
        '--max_batches', str(args.max_batches),
        '--batch_size', str(args.batch_size),
        '--log_dir', args.log_dir,
        '--device', args.device,
    ]

    print("=" * 70)
    print("Running Robustness Evaluation")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    # Run the evaluation
    script_start_time = time.time()

    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        output_lines = []
        for line in process.stdout:
            print(line, end='')
            output_lines.append(line)

        process.wait()
        output = ''.join(output_lines)

        if process.returncode != 0:
            print(f"\nEvaluation failed with return code: {process.returncode}")
            sys.exit(process.returncode)

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        process.terminate()
        sys.exit(1)
    except Exception as e:
        print(f"\nError running evaluation: {e}")
        sys.exit(1)

    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time

    # Parse wall times from output
    wall_times = parse_wall_times(output)

    # Print wall time summary
    if wall_times:
        print_wall_time_summary(wall_times, total_script_time)
    else:
        print(f"\nTotal script execution time: {format_time(total_script_time)}")

    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
