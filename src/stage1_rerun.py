"""Helpers for re-running Stage 1 on combinations that already have results.

Used by the `restage1` mode of submit_miraca_uncertainty_study.sh, one call per
combination inside its job, and by the gate that holds Stage 2 back:

  backup          copy the Stage 1 files the rerun rewrites (segments, meta,
                  validation marker, the listed hazards' profiles) to a backup
                  folder, then delete the live validation marker. Never
                  overwrites: a resubmitted job finds its backup done and
                  leaves it alone, so the backup stays the original.
  mark-validated  write the validation marker run_study writes after a pass.
  check-segments  compare the rebuilt segments with the backup. Every column must
                  be identical except those named with --allow, which may also be
                  new (the river protection prot_rp for the 5 km blocks, and its
                  provenance columns prot_source, prot_fill_km), and those named
                  with --allow-unexposed (coast_prot_rp), which may change only on
                  features no coastal profile reaches (present day or any SLR
                  state), where the value enters no result. Anything else that
                  moved would invalidate results the rerun does not redo, e.g.
                  coastal ones, so it fails the job.
  extracted       exit 0 if this combination's listed hazards were already
                  extracted after its backup, by the DamageScanner its asset type
                  requires; the rerun job then skips preprocess (resubmitting a
                  combination that failed only a later check costs minutes).
  gate            for a whole list: backup taken, segments check passed, every
                  rerun hazard extracted by the required DamageScanner, and a
                  validation newer than the extraction. Exit 1 if anything is
                  missing, which blocks the Stage 2 jobs that depend on it.
                  --write-failed FILE lists the combinations that are not ready.

    python -m src.stage1_rerun backup --country DEU --asset power --hazards river earthquake windstorm \\
        --dest data/intermediate_ds101_backup_20261120
    python -m src.stage1_rerun check-segments --country DEU --asset power --allow prot_rp prot_source prot_fill_km \\
        --backup data/intermediate_ds101_backup_20261120 --status reruns/20261120_polygon_fix/status
    python -m src.stage1_rerun gate --list reruns/20261120_polygon_fix/stage1_combos.txt \\
        --hazards river earthquake windstorm --backup ... --status ... --require-polygon-fix
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import ds_version
from .curves import applicable_hazards
from .paths import PROJECT_ROOT, base_stem, hazard_stem, load_config, set_asset_override, set_country_override


def _cfg(country: str, asset: str) -> dict:
    set_country_override(country)
    set_asset_override(asset)
    return load_config()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _files(cfg: dict, hazards: list[str]) -> tuple[list[Path], list[Path]]:
    """(required, optional) Stage 1 files of one combination that a rerun rewrites."""
    idir, stem = Path(cfg["intermediate_dir"]), base_stem(cfg)
    required = [idir / f"{stem}_segments.parquet", idir / f"{stem}_meta.json"]
    optional = [idir / f"{stem}_validated.ok"]
    optional += [idir / f"{hazard_stem(cfg, h)}_profiles.parquet" for h in hazards]
    return required, optional


def _done_flag(dest: Path, cfg: dict) -> Path:
    return dest / f"{base_stem(cfg)}.backup_done"


def backup(country: str, asset: str, hazards: list[str], dest: Path) -> None:
    cfg = _cfg(country, asset)
    flag = _done_flag(dest, cfg)
    marker = Path(cfg["intermediate_dir"]) / f"{base_stem(cfg)}_validated.ok"
    if flag.exists():
        print(f"{country} {asset}: backup already taken ({flag.read_text().splitlines()[0]}); left as it is")
        marker.unlink(missing_ok=True)      # Stage 1 is redone, so its validation is too
        return
    dest.mkdir(parents=True, exist_ok=True)
    required, optional = _files(cfg, hazards)
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"{country} {asset}: nothing to back up, missing {missing}")
    copied = []
    for src in required + [p for p in optional if p.exists()]:
        dst = dest / src.name
        if dst.exists():
            if filecmp.cmp(src, dst, shallow=False):
                continue
            raise SystemExit(f"{dst} exists and differs from {src}; not overwriting a backup")
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        copied.append(src.name)
    flag.write_text(datetime.now(timezone.utc).isoformat() + "\n" + "\n".join(copied) + "\n")
    marker.unlink(missing_ok=True)
    print(f"{country} {asset}: backed up {len(copied)} file(s) to {dest}")


def mark_validated(country: str, asset: str) -> None:
    """The marker run_study writes after a passed validation, so Stage 2 skips it."""
    cfg = _cfg(country, asset)
    marker = Path(cfg["intermediate_dir"]) / f"{base_stem(cfg)}_validated.ok"
    marker.write_text(datetime.now(timezone.utc).isoformat())
    print(f"{country} {asset}: validated, wrote {marker.name}")


def _same(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Element-wise equality, NaN equal to NaN, exact for numbers."""
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        x, y = a.to_numpy(np.float64), b.to_numpy(np.float64)
        return (x == y) | (np.isnan(x) & np.isnan(y))
    x, y = a.astype("string"), b.astype("string")
    return ((x == y).fillna(False) | (x.isna() & y.isna())).to_numpy(bool)


def _coastal_exposed(cfg: dict, rows: np.ndarray) -> np.ndarray:
    """Which of `rows` any coastal profile (present day or an SLR state) reaches."""
    if len(rows) == 0:
        return np.zeros(0, bool)
    hit = set()
    idir, stem = Path(cfg["intermediate_dir"]), base_stem(cfg)
    for p in sorted(idir.glob(f"{stem}_coastal*_profiles.parquet")):
        t = pq.read_table(p, columns=["seg", "intensity"], filters=[("seg", "in", rows.tolist())]).to_pandas()
        hit |= set(t.seg[t.intensity > 0].tolist())
    return np.isin(rows, list(hit))


def check_segments(country: str, asset: str, backup_dir: Path, allow: list[str], status_dir: Path,
                   allow_unexposed: list[str] | None = None) -> bool:
    allow_unexposed = allow_unexposed or []
    cfg = _cfg(country, asset)
    name = f"{base_stem(cfg)}_segments.parquet"
    new = pd.read_parquet(Path(cfg["intermediate_dir"]) / name)
    old = pd.read_parquet(backup_dir / name)
    report = {"country": country, "asset": asset, "rows_old": len(old), "rows_new": len(new),
              "checked": datetime.now(timezone.utc).isoformat(), "problems": [], "allowed_changes": {},
              "allowed_unexposed_changes": {}}
    if len(old) != len(new):
        report["problems"].append(f"row count {len(old)} -> {len(new)}")
    gone = sorted(set(old.columns) - set(new.columns))
    added = sorted(set(new.columns) - set(old.columns))
    if gone or set(added) - set(allow):
        report["problems"].append(f"columns differ: only old {gone}, only new {added}")
    report["added_columns"] = added           # allowed ones, e.g. the river protection provenance
    if not report["problems"]:
        for col in old.columns:
            same = _same(old[col].reset_index(drop=True), new[col].reset_index(drop=True))
            n = int((~same).sum())
            if col in allow:
                report["allowed_changes"][col] = n
            elif n and col in allow_unexposed:
                rows = np.flatnonzero(~same)
                exposed = rows[_coastal_exposed(cfg, rows)]
                report["allowed_unexposed_changes"][col] = int(len(rows) - len(exposed))
                if len(exposed):
                    report["problems"].append(f"{col}: {len(exposed)} coastally exposed row(s) changed, "
                                              f"e.g. {exposed[:5].tolist()}")
            elif n:
                report["problems"].append(f"{col}: {n} of {len(old)} rows changed")
    report["ok"] = not report["problems"]
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / f"{base_stem(cfg)}.segments.json").write_text(json.dumps(report, indent=2))
    verdict = "OK" if report["ok"] else "FAILED: " + "; ".join(report["problems"])
    print(f"{country} {asset}: segments vs backup {verdict}; allowed changes {report['allowed_changes']}"
          + (f"; on rows without coastal exposure {report['allowed_unexposed_changes']}" if report["allowed_unexposed_changes"] else ""))
    return report["ok"]


def _here(cfg: dict, asset: str, country: str, hazards: list[str]) -> list[str]:
    return [h for h in hazards if h in applicable_hazards(asset, country) and h in cfg["hazards"]]


def extracted(country: str, asset: str, hazards: list[str], backup_dir: Path) -> bool:
    """True if every listed hazard was extracted after the backup, by an acceptable DamageScanner."""
    cfg = _cfg(country, asset)
    stem, idir = base_stem(cfg), Path(cfg["intermediate_dir"])
    flag, meta_p = _done_flag(backup_dir, cfg), idir / f"{stem}_meta.json"
    why = []
    if not flag.exists():
        why.append("no backup yet")
    elif not meta_p.exists() or meta_p.stat().st_mtime <= flag.stat().st_mtime:
        why.append("not extracted since the backup")
    else:
        meta = json.loads(meta_p.read_text())
        for h in _here(cfg, asset, country, hazards):
            ds = (meta.get("hazards", {}).get(h) or {}).get("damagescanner")
            if not (idir / f"{hazard_stem(cfg, h)}_profiles.parquet").exists():
                why.append(f"{h}: no profiles")
            elif not ds:
                why.append(f"{h}: not re-extracted")
            elif ds_version.problem(asset, ds):
                why.append(f"{h}: extracted with the wrong DamageScanner")
    print(f"{country} {asset}: " + ("extraction done, preprocess not needed" if not why
                                    else "preprocess needed (" + "; ".join(why) + ")"))
    return not why


def read_list(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            c, a = line.split()[:2]
            out.append((c, a))
    return out


def gate(combos, hazards, backup_dir: Path, status_dir: Path, require_fix: bool,
         write_failed: Path | None = None) -> bool:
    rows = []
    for country, asset in combos:
        cfg = _cfg(country, asset)
        stem, idir = base_stem(cfg), Path(cfg["intermediate_dir"])
        problems = []
        if not _done_flag(backup_dir, cfg).exists():
            problems.append("no backup")
        seg = status_dir / f"{stem}.segments.json"
        if not seg.exists():
            problems.append("segments not checked")
        elif not json.loads(seg.read_text())["ok"]:
            problems.append("segments differ from backup")
        meta_p = idir / f"{stem}_meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        for h in _here(cfg, asset, country, hazards):
            ds = (meta.get("hazards", {}).get(h) or {}).get("damagescanner")
            if not ds:
                problems.append(f"{h}: not re-extracted")
            elif require_fix and not (ds.get("polygon_fix") and ds.get("commit") == ds_version.POLYGON_FIX_COMMIT):
                problems.append(f"{h}: extracted without the pinned polygon fix")
        marker = idir / f"{stem}_validated.ok"
        if not marker.exists():
            problems.append("not validated")
        elif meta_p.exists() and marker.stat().st_mtime < meta_p.stat().st_mtime:
            problems.append("validation older than the extraction")
        rows.append({"country": country, "asset": asset, "problems": "; ".join(problems)})
    df = pd.DataFrame(rows)
    bad = df[df.problems != ""]
    print(f"{len(df) - len(bad)} of {len(df)} combinations ready for Stage 2")
    if len(bad):
        print(bad.to_string(index=False))
    out = status_dir / "gate.csv"
    status_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    if write_failed is not None:
        write_failed.write_text("".join(f"{c} {a}\n" for c, a in bad[["country", "asset"]].itertuples(index=False)),
                                newline="\n")
        print(f"wrote {len(bad)} combination(s) not ready to {write_failed}")
    return bad.empty


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backup")
    b.add_argument("--country", required=True)
    b.add_argument("--asset", required=True)
    b.add_argument("--hazards", nargs="+", required=True)
    b.add_argument("--dest", required=True)
    c = sub.add_parser("check-segments")
    c.add_argument("--country", required=True)
    c.add_argument("--asset", required=True)
    c.add_argument("--backup", required=True)
    c.add_argument("--allow", nargs="*", default=[])
    c.add_argument("--allow-unexposed", nargs="*", default=[])
    c.add_argument("--status", required=True)
    e = sub.add_parser("extracted")
    e.add_argument("--country", required=True)
    e.add_argument("--asset", required=True)
    e.add_argument("--hazards", nargs="+", required=True)
    e.add_argument("--backup", required=True)
    m = sub.add_parser("mark-validated")
    m.add_argument("--country", required=True)
    m.add_argument("--asset", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--list", required=True)
    g.add_argument("--hazards", nargs="+", required=True)
    g.add_argument("--backup", required=True)
    g.add_argument("--status", required=True)
    g.add_argument("--require-polygon-fix", action="store_true")
    g.add_argument("--write-failed", default=None)
    args = ap.parse_args()

    if args.cmd == "backup":
        backup(args.country, args.asset, args.hazards, _resolve(args.dest))
    elif args.cmd == "mark-validated":
        mark_validated(args.country, args.asset)
    elif args.cmd == "check-segments":
        ok = check_segments(args.country, args.asset, _resolve(args.backup), args.allow, _resolve(args.status),
                            args.allow_unexposed)
        sys.exit(0 if ok else 1)
    elif args.cmd == "extracted":
        sys.exit(0 if extracted(args.country, args.asset, args.hazards, _resolve(args.backup)) else 1)
    else:
        ok = gate(read_list(_resolve(args.list)), args.hazards, _resolve(args.backup), _resolve(args.status),
                  args.require_polygon_fix, _resolve(args.write_failed) if args.write_failed else None)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
