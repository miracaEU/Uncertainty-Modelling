"""Copy the current per-combination results aside before a rerun overwrites them.

The raw experiment archives already survive a rerun: they are named
``experiments_{...}_sobol_n{N}_{timestamp}.tar.gz``, so a run at a different N
lands beside the old one rather than on it, and the aggregator pairs the two
into its precision comparison. What does NOT survive is the derived output -
``{country}_{asset}_{scenario}_sobol_indices.csv``, ``_feature_scores.csv`` and
the per-combination figures all live at fixed paths and are rewritten in place.

This copies those aside first. The destination is deliberately OUTSIDE
``results/``: aggregate_results.py discovers combinations with
``results_dir.rglob("*_sobol_indices.csv")`` and parses the scenario out of the
filename, so a snapshot kept inside - under any name - would either be read as a
duplicate of the live combination or invent a phantom scenario.

    python -m src.snapshot_results --list sobol_retry/_all_nonconverged.csv --tag n8192
    python -m src.snapshot_results --list ... --tag n8192 --dry-run
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path

import pandas as pd

from .paths import load_config

# Everything written per combination that a rerun would overwrite. The archives
# are excluded on purpose - they are already versioned by N and timestamp, and
# copying them would double tens of GB to no end.
PATTERNS = (
    "{c}_{a}_{s}_sobol_indices.csv",
    "{c}_{a}_{s}_feature_scores.csv",
    "figures/{c}_{a}_{s}_sobol_indices.png",
    "figures/{c}_{a}_{s}_feature_scores.png",
    "figures/{c}_{a}_{s}_ead_vs_protection.png",
    "figures/{c}_{a}_{s}_ead_vs_depth_scale.png",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True,
                    help="CSV with country/asset/scenario columns naming the "
                         "combinations to snapshot.")
    ap.add_argument("--tag", required=True,
                    help="suffix for the snapshot directory, e.g. n8192 -> "
                         "results_snapshot_n8192/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    results = Path(cfg["results_dir"])
    dest_root = results.with_name(f"{results.name}_snapshot_{args.tag}")

    combos = pd.read_csv(args.list)
    missing_cols = {"country", "asset", "scenario"} - set(combos.columns)
    if missing_cols:
        raise SystemExit(f"{args.list} lacks column(s): {sorted(missing_cols)}")

    copied = skipped = same = 0
    clash = []
    for row in combos.to_dict("records"):
        c, a, s = row["country"], row["asset"], row["scenario"]
        for pat in PATTERNS:
            rel = Path(c) / pat.format(c=c, a=a, s=s)
            src = results / rel
            if not src.exists():
                skipped += 1
                continue
            dst = dest_root / rel
            # A snapshot is never overwritten: an identical copy is already
            # safe, and a different one is an older state worth keeping.
            if dst.exists():
                if filecmp.cmp(src, dst, shallow=False):
                    same += 1
                else:
                    clash.append(dst)
                continue
            if not args.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            copied += 1

    verb = "would copy" if args.dry_run else "copied"
    print(f"{verb} {copied} file(s) for {len(combos)} combination(s) -> {dest_root}")
    if same:
        print(f"{same} file(s) already in the snapshot, identical")
    if clash:
        print(f"!! {len(clash)} file(s) already in the snapshot with DIFFERENT content - left as "
              "they are, and the live file NOT backed up. Use another --tag:")
        for p in clash:
            print(f"   {p}")
        raise SystemExit(1)
    print(f"{skipped} expected file(s) absent (not every combination writes every figure)")
    if not args.dry_run:
        print("\nThe experiment archives were NOT copied - they are already "
              "versioned by N and timestamp and survive the rerun in place.")


if __name__ == "__main__":
    main()
