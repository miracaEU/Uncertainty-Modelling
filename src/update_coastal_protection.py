"""Refresh coast_prot_rp in existing Stage 1 segments, without touching anything else.

Recomputes each feature's coastal design standard with src.coastal's current rules
(sample_coastal_protection) and compares it with the stored column. Written for
the minimum standard of November 2026 (coastal.MINIMUM_STANDARD: Malta, Cyprus),
which changes only features that were at 0; any other change is reported and
stops --apply for that combination.

  python -m src.update_coastal_protection --out reruns/<dir> --countries MLT CYP            # report
  python -m src.update_coastal_protection --out reruns/<dir> --countries MLT CYP \\
      --apply --backup-dir data/intermediate_coastal_backup_<date>

--apply copies the segments and meta files to --backup-dir first (never
overwriting a backup), then rewrites coast_prot_rp through a temporary file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .coastal import MINIMUM_STANDARD, sample_coastal_protection
from .paths import PROJECT_ROOT, load_config, set_asset_override, set_country_override
from .preprocess import classify_geometry, drop_unmapped_object_types, load_exposure

TOL = 1e-3


def one(country: str, asset: str, apply: bool, backup_dir: Path | None) -> dict:
    set_country_override(country)
    set_asset_override(asset)
    cfg = load_config()
    idir = cfg["intermediate_dir"]
    seg_p, meta_p = idir / f"{country}_{asset}_segments.parquet", idir / f"{country}_{asset}_meta.json"
    row = {"country": country, "asset": asset}
    try:
        seg = pd.read_parquet(seg_p)
        feats = load_exposure(cfg)
        feats, _ = drop_unmapped_object_types(feats, classify_geometry(feats), cfg)
        if len(feats) != len(seg) or not (feats["osm_id"].to_numpy() == seg["osm_id"].to_numpy()).all():
            raise ValueError("segments and exposure rows do not line up")
        new = sample_coastal_protection(feats, cfg["hazards"]["coastal"]).astype(float)
        old = seg["coast_prot_rp"].to_numpy(float)
        changed = np.abs(new - old) > TOL
        prof = idir / f"{country}_{asset}_coastal_profiles.parquet"
        exposed = np.zeros(len(seg), bool)
        if prof.exists():
            p = pd.read_parquet(prof, columns=["seg", "intensity"])
            exposed[np.unique(p.seg.to_numpy()[p.intensity.to_numpy() > 0])] = True
        unexpected = changed & (old > 0)
        row.update({"n_features": len(seg), "n_exposed": int(exposed.sum()), "n_changed": int(changed.sum()),
                    "n_exposed_changed": int((changed & exposed).sum()),
                    "n_changed_from_nonzero": int(unexpected.sum()),
                    "new_values": ",".join(f"{v:g}" for v in sorted(set(np.round(new[changed], 3)))),
                    "status": "ok"})
        if apply:
            if unexpected.any():
                raise ValueError(f"{int(unexpected.sum())} feature(s) with a standard above 0 would change")
            if not changed.any():
                row["status"] = "unchanged"
                return row
            backup_dir.mkdir(parents=True, exist_ok=True)
            for src in (seg_p, meta_p):
                if src.exists():
                    dst = backup_dir / src.name
                    if dst.exists():
                        raise FileExistsError(f"backup {dst} exists; not overwriting it")
                    shutil.copy2(src, dst)
            out = seg.copy()
            out["coast_prot_rp"] = new.astype(np.float32)
            tmp = seg_p.with_suffix(".parquet.tmp")
            out.to_parquet(tmp, index=False)
            os.replace(tmp, seg_p)
            meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
            meta["coastal_minimum_standard"] = MINIMUM_STANDARD
            meta["coastal_protection_updated"] = datetime.now(timezone.utc).isoformat()
            tmp = meta_p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            os.replace(tmp, meta_p)
            row["status"] = "applied"
    except Exception as exc:                                   # noqa: BLE001
        row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--countries", nargs="+", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backup-dir", default=None)
    args = ap.parse_args()
    if args.apply and not args.backup_dir:
        raise SystemExit("--apply needs --backup-dir")
    cfg = load_config()
    rows = []
    for f in sorted(Path(cfg["intermediate_dir"]).glob("*_segments.parquet")):
        c, a = re.match(r"([A-Z]{3})_([a-z]+)_segments", f.name).groups()
        if c not in args.countries or "coast_prot_rp" not in pd.read_parquet(f, columns=None).columns:
            continue
        rows.append(one(c, a, args.apply, PROJECT_ROOT / args.backup_dir if args.backup_dir else None))
        r = rows[-1]
        print(f"{c} {a}: {r['status']} {r.get('n_exposed_changed', '')}/{r.get('n_exposed', '')} exposed changed "
              f"-> {r.get('new_values', '')} {r.get('error', '')}", flush=True)
    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    name = "coastal_protection_applied.csv" if args.apply else "coastal_protection_changes.csv"
    pd.DataFrame(rows).to_csv(out / name, index=False)
    print("wrote", out / name)
    bad = [r for r in rows if r["status"] == "error"]
    if bad:
        raise SystemExit(f"{len(bad)} combination(s) failed")


if __name__ == "__main__":
    main()
