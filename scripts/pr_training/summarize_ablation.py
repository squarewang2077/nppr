# summarize_ablation.py - Turn a directory of training CSVs into one comparison
#                         table, and optionally splice it into a markdown file.
#
# Description:
#   Reads every *_training_info.csv under a directory, takes each run's final
#   evaluated epoch, and prints a markdown table ranked by robust accuracy.
#
#   Two CSV schemas are read, because the ablation compares runs from two
#   trainers that name their columns differently and neither is worth renaming
#   (existing notebooks and eval scripts read the current names):
#
#     pos_geo_training.py    test_clean  test_pgd  test_fgsm  test_pr_laplace
#     train_classifiers.py   val_acc     val_pgd   val_fgsm   val_random_l
#
#   Evaluation runs every --eval_every epochs, so the last row of a CSV usually
#   has empty evaluation columns. The last row where clean accuracy is present
#   is used instead — that is the run's final measured result.
#
#   The three diagnostic columns on the right are a validity gate, not a
#   result. valid_position_rate is the fraction of perturbations that actually
#   reached the level set; when it is low the geometry columns describe points
#   that are not on the level being studied, and that row says nothing about
#   its configuration. Likewise ae_rate_valid below 1.0 means some weighted
#   positions were not adversarial examples at all. Read those before reading
#   the accuracy columns.
#
# Requirements:
#   pandas
#
# Usage:
#   python scripts/pr_training/summarize_ablation.py <dir> [--out <markdown>]
#
# Examples:
#   # Print the screening table
#   python scripts/pr_training/summarize_ablation.py \
#       results/pos_geo_training/resnet18_cifar10/screen
#
#   # Splice it into the experiment's markdown, leaving the prose alone
#   python scripts/pr_training/summarize_ablation.py \
#       results/pos_geo_training/resnet18_cifar10/screen \
#       --out run_exp/pos_geo_training/run_resnet18_on_cifar10.md

import argparse
import glob
import os
import re

import pandas as pd

# Metric -> the column names it may go by, most specific first.
METRIC_ALIASES = {
    "test_clean":      ("test_clean", "val_acc"),
    "test_pgd":        ("test_pgd", "val_pgd"),
    "test_fgsm":       ("test_fgsm", "val_fgsm"),
    "test_pr_laplace": ("test_pr_laplace", "val_random_l"),
    "trainS_clean":    ("trainS_clean", "trainS_acc"),
    "trainS_pgd":      ("trainS_pgd",),
}

# Diagnostics averaged over the run's epochs. Absent for baseline runs, which
# have no level set to be on.
DIAG_MEANS = ("valid_position_rate", "ae_rate_valid", "eff_rank")

# Per-epoch wall clock, named differently by the two trainers.
TIME_ALIASES = ("train_time_s", "time")

BEGIN_FMT = "<!-- BEGIN AUTO:{name} -->"
END_MARK = "<!-- END AUTO -->"


def _col(df, aliases):
    """First alias present in *df*, or None."""
    for name in aliases:
        if name in df.columns:
            return name
    return None


def _get(row, df, aliases):
    col = _col(df, aliases)
    if col is None:
        return None
    value = row[col]
    return None if pd.isna(value) else float(value)


def _config_summary(row, df):
    """One-line description of what this run was, from the CSV's own columns."""
    if "training_type" in df.columns and not pd.isna(row.get("training_type")):
        return str(row["training_type"])

    parts = []
    for key in ("solver", "weight_mode", "geometry_mode"):
        if key in df.columns and not pd.isna(row.get(key)):
            parts.append(str(row[key]))
    # geometry_mode is only read by sharp/flat; showing it elsewhere invites
    # the reader to attribute a difference to a knob that did nothing.
    if len(parts) == 3 and parts[1] not in ("sharp", "flat"):
        parts.pop()

    t_mode = row.get("t_mode") if "t_mode" in df.columns else None
    if t_mode == "reachable" and "t_frac" in df.columns:
        parts.append(f"t_frac={row['t_frac']:g}")
    elif t_mode == "fixed" and "t" in df.columns:
        parts.append(f"t={row['t']:g}")
    elif t_mode is not None and not pd.isna(t_mode):
        parts.append(str(t_mode))

    for key, label in (("num_starts", "N"), ("num_steps", "steps")):
        if key in df.columns and not pd.isna(row.get(key)):
            parts.append(f"{label}={int(row[key])}")
    return " ".join(parts)


def read_run(path):
    """One summary dict for one CSV, or None if it has no evaluated epoch."""
    df = pd.read_csv(path)
    if df.empty:
        return None

    clean_col = _col(df, METRIC_ALIASES["test_clean"])
    if clean_col is None:
        return None
    evaluated = df[df[clean_col].notna()]
    if evaluated.empty:
        return None
    last = evaluated.iloc[-1]

    # The tag column exists only on newer pos_geo runs; fall back to the
    # filename, which is where the tag ended up anyway.
    tag = last["tag"] if "tag" in df.columns and not pd.isna(last.get("tag")) else \
        os.path.basename(path).replace("_training_info.csv", "")

    out = {
        "tag": str(tag),
        "config": _config_summary(last, df),
        "epoch": int(last["epoch"]),
        "is_baseline": "training_type" in df.columns,
    }
    for metric, aliases in METRIC_ALIASES.items():
        out[metric] = _get(last, df, aliases)

    for key in DIAG_MEANS:
        out[key] = float(df[key].mean()) if key in df.columns else None

    time_col = _col(df, TIME_ALIASES)
    out["time_s"] = float(df[time_col].mean()) if time_col else None
    return out


def _pct(value):
    return "—" if value is None else f"{value * 100:.2f}"


def _num(value, fmt=".3f"):
    return "—" if value is None else f"{value:{fmt}}"


def render(rows):
    """Markdown table: baselines first, then ablation runs by robust accuracy."""
    header = (
        "| run | config | clean | PGD-10 | FGSM | PR(lap) | clean−PGD | "
        "valid | ae | eff_rank | s/ep |\n"
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n"
    )

    def line(r):
        gap = (None if r["test_clean"] is None or r["test_pgd"] is None
               else r["test_clean"] - r["test_pgd"])
        return (f"| `{r['tag']}` | {r['config']} | {_pct(r['test_clean'])} | "
                f"**{_pct(r['test_pgd'])}** | {_pct(r['test_fgsm'])} | "
                f"{_pct(r['test_pr_laplace'])} | {_pct(gap)} | "
                f"{_num(r['valid_position_rate'])} | {_num(r['ae_rate_valid'])} | "
                f"{_num(r['eff_rank'], '.2f')} | {_num(r['time_s'], '.0f')} |\n")

    # Sort by robust accuracy; a run with no PGD number sinks rather than
    # crashing the comparison.
    def key(r):
        return -1.0 if r["test_pgd"] is None else r["test_pgd"]

    baselines = sorted([r for r in rows if r["is_baseline"]], key=key, reverse=True)
    ablation = sorted([r for r in rows if not r["is_baseline"]], key=key, reverse=True)

    out = ""
    if baselines:
        out += "**Baselines** (same optimiser, epochs and evaluation)\n\n"
        out += header + "".join(line(r) for r in baselines) + "\n"
    if ablation:
        out += "**Ablation** (each row differs from `ref` in one thing)\n\n"
        out += header + "".join(line(r) for r in ablation) + "\n"

    epochs = sorted({r["epoch"] for r in rows})
    out += (f"Accuracies are percentages on the test set at epoch "
            f"{epochs[0] if len(epochs) == 1 else f'{epochs[0]}–{epochs[-1]}'}. "
            f"`valid` / `ae` / `eff_rank` are averaged over every epoch of the "
            f"run. **Read `valid` and `ae` first**: a run whose perturbations "
            f"did not reach the level set, or whose landings were not "
            f"adversarial examples, is not evidence about its configuration.\n")
    return out


def splice(md_path, name, table):
    """Replace the AUTO block named *name* in *md_path*, leaving prose alone."""
    begin = BEGIN_FMT.format(name=name)
    text = open(md_path, encoding="utf-8").read() if os.path.exists(md_path) else ""

    block = f"{begin}\n\n{table}\n{END_MARK}"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(END_MARK), re.DOTALL)
    if pattern.search(text):
        text = pattern.sub(lambda _: block, text)
    else:
        # No placeholder to fill: append rather than silently dropping the
        # table, and say so, since the intended spot was probably mistyped.
        print(f"note: {begin} not found in {md_path}, appending a new block")
        text = text.rstrip() + f"\n\n## {name}\n\n{block}\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser(
        description="Summarise a directory of training CSVs into one table.")
    ap.add_argument("results_dir",
                    help="Directory holding *_training_info.csv files.")
    ap.add_argument("--out", default=None,
                    help="Markdown file to splice the table into, between "
                         "<!-- BEGIN AUTO:<stage> --> and <!-- END AUTO -->. "
                         "Prints to stdout when omitted.")
    ap.add_argument("--name", default=None,
                    help="Name of the AUTO block to replace. Defaults to the "
                         "results directory's own name, e.g. 'screen'.")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results_dir, "*_training_info.csv")))
    if not paths:
        raise SystemExit(f"no *_training_info.csv under {args.results_dir}")

    rows = []
    for path in paths:
        row = read_run(path)
        if row is None:
            print(f"skipping {os.path.basename(path)}: no evaluated epoch yet")
            continue
        rows.append(row)
    if not rows:
        raise SystemExit("every CSV was empty or had no evaluated epoch")

    table = render(rows)
    name = args.name or os.path.basename(os.path.normpath(args.results_dir))

    if args.out:
        splice(args.out, name, table)
        print(f"wrote {len(rows)} runs into {args.out} (block '{name}')")
    else:
        print(table)


if __name__ == "__main__":
    main()
