"""Switch existing Stage 1 segments to the configured river protection method.

The river protection standard (prot_rp) is a column of each
{country}_{asset}_segments.parquet, written by Stage 1 with whatever method the
config named then - until November 2026 the 500 m FLOPROS pixel. This recomputes
it with src.river_protection (5 km block means, 0-blocks filled within fill_km)
without touching anything else in the table or any hazard profile, which
preprocess --segments-only would also rebuild (and re-sample COASTPROS).

For every combination with river profiles it reports how many river-EXPOSED
features (any profile row with depth > 0) change standard. Only those features
enter the river EAD, so a combination with none changed keeps valid results
under every scenario that reads prot_rp.

  python -m src.update_river_protection --out reruns/<dir>               # report only
  python -m src.update_river_protection --out reruns/<dir> --assets roads rail telecom \\
      --apply --backup-dir data/intermediate_pixel_backup_<date>

--apply first copies the segments and meta files to --backup-dir (refusing to
overwrite a backup that exists), then replaces prot_rp and adds prot_source and
prot_fill_km, writing to a temporary file that is renamed over the original.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
import shapely

from .curves import get_asset_config, is_mapped
from .paths import PROJECT_ROOT, load_config
from .preprocess import river_protection_settings
from .river_protection import FROM_FILL, assign_river_protection

CHANGE_TOL = 1e-3   # years; prot_rp is stored as float32


def centroids(path: Path, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """EPSG:3035 centroid x, y of the kept exposure rows, streamed in batches."""
    pf = pq.ParquetFile(path)
    crs = (json.loads(pf.schema_arrow.metadata[b"geo"])["columns"]["geometry"].get("crs") or {})
    epsg = (crs.get("id") or {}).get("code") if isinstance(crs, dict) else None
    if epsg not in (None, 3035):
        raise ValueError(f"{path.name}: geometry is EPSG:{epsg}, expected 3035")
    xs, ys, start = [], [], 0
    for batch in pf.iter_batches(columns=["geometry"], batch_size=200_000):
        n = batch.num_rows
        take = keep[(keep >= start) & (keep < start + n)] - start
        if take.size:
            g = shapely.from_wkb(batch.column(0).take(take).to_numpy(zero_copy_only=False))
            c = shapely.centroid(g)
            xs.append(shapely.get_x(c))
            ys.append(shapely.get_y(c))
        start += n
    return np.concatenate(xs), np.concatenate(ys)


def pixel_values(path: Path, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The 500 m pixel under each point, 0 for no data: the old method, for a check."""
    with rasterio.open(path) as r:
        rows, cols = rasterio.transform.rowcol(r.transform, x, y)
        rows, cols = np.asarray(rows), np.asarray(cols)
        ok = (rows >= 0) & (rows < r.height) & (cols >= 0) & (cols < r.width)
        out = np.zeros(len(x))
        if ok.any():
            r0, r1, c0, c1 = rows[ok].min(), rows[ok].max() + 1, cols[ok].min(), cols[ok].max() + 1
            win = r.read(1, window=rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0))
            v = win[rows[ok] - r0, cols[ok] - c0].astype(float)
            out[ok] = np.where((v == r.nodata) | ~np.isfinite(v), 0.0, v)
    return np.clip(out, 0, None)


def one(task: dict) -> dict:
    cfg = load_config()
    c, a = task["country"], task["asset"]
    idir = cfg["intermediate_dir"]
    seg_p, prof_p = idir / f"{c}_{a}_segments.parquet", idir / f"{c}_{a}_river_profiles.parquet"
    meta_p = idir / f"{c}_{a}_meta.json"
    row = {"country": c, "asset": a}
    try:
        seg = pd.read_parquet(seg_p)
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        row["method_before"] = (meta.get("river_protection") or {}).get("method", "pixel")
        expo = Path(cfg["exposure_dir"]) / f"{c}_{a}_exposure.parquet"
        ids = pq.read_table(expo, columns=["osm_id", "object_type"]).to_pandas()
        keep = np.flatnonzero(np.asarray(is_mapped(get_asset_config(a), ids.object_type)))
        if len(keep) != len(seg) or not (ids.osm_id.to_numpy()[keep] == seg.osm_id.to_numpy()).all():
            raise ValueError("segments and exposure rows do not line up")
        x, y = centroids(expo, keep)
        rp = river_protection_settings(cfg)
        new = assign_river_protection(x, y, cfg["protection_standard_path"], rp.get("fill_km", 5.0))
        prof = pq.read_table(prof_p, columns=["seg", "intensity"]).to_pandas()
        exposed = np.zeros(len(seg), bool)
        exposed[np.unique(prof.seg.to_numpy()[prof.intensity.to_numpy() > 0])] = True
        old = seg["prot_rp"].to_numpy(float)
        newv = new.prot_rp.to_numpy(np.float32).astype(float)
        changed = np.abs(newv - old) > CHANGE_TOL
        pix = pixel_values(cfg["protection_standard_path"], x[exposed], y[exposed])
        rel = np.abs(newv - old)[exposed & changed] / np.where(old[exposed & changed] > 0,
                                                               old[exposed & changed], np.nan)
        row.update({
            "n_features": len(seg), "n_exposed": int(exposed.sum()),
            "n_exposed_changed": int((exposed & changed).sum()),
            "n_exposed_zero_before": int((exposed & (old == 0)).sum()),
            "n_exposed_zero_after": int((exposed & (newv == 0)).sum()),
            "n_exposed_filled": int((exposed & (new.source.to_numpy() == FROM_FILL)).sum()),
            "median_rel_change": float(np.nanmedian(rel)) if np.isfinite(rel).any() else np.nan,
            "max_abs_change": float(np.abs(newv - old)[exposed].max()) if exposed.any() else 0.0,
            "n_features_changed": int(changed.sum()),
            "old_vs_pixel_mismatch": int((np.abs(pix - old[exposed]) > CHANGE_TOL).sum()),
            "status": "ok",
        })
        if task["apply"]:
            if row["method_before"] == rp["method"]:
                row["status"] = "already applied"
                return row
            bdir = Path(task["backup_dir"])
            bdir.mkdir(parents=True, exist_ok=True)
            for p in (seg_p, meta_p):
                if p.exists():
                    dst = bdir / p.name
                    if dst.exists():
                        raise FileExistsError(f"backup {dst} exists; not overwriting it")
                    shutil.copy2(p, dst)
            out = seg.copy()
            out["prot_rp"] = newv.astype(np.float32)
            out["prot_source"] = new.source.to_numpy(np.int8)
            out["prot_fill_km"] = new.fill_km.to_numpy(np.float32)
            tmp = seg_p.with_suffix(".parquet.tmp")
            out.to_parquet(tmp, index=False)
            os.replace(tmp, seg_p)
            meta["river_protection"] = rp
            meta["river_protection_updated"] = datetime.now(timezone.utc).isoformat()
            tmp = meta_p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            os.replace(tmp, meta_p)
            row["status"] = "applied"
    except Exception as exc:                                   # noqa: BLE001
        row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="folder for river_protection_changes.csv")
    ap.add_argument("--assets", nargs="*", default=None)
    ap.add_argument("--countries", nargs="*", default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backup-dir", default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.apply and not args.backup_dir:
        raise SystemExit("--apply needs --backup-dir")

    cfg = load_config()
    print("method:", river_protection_settings(cfg))
    tasks = []
    for f in sorted(Path(cfg["intermediate_dir"]).glob("*_river_profiles.parquet")):
        c, a = re.match(r"([A-Z]{3})_([a-z]+)_river_profiles", f.name).groups()
        if (args.assets and a not in args.assets) or (args.countries and c not in args.countries):
            continue
        tasks.append({"country": c, "asset": a, "apply": args.apply,
                      "backup_dir": str(PROJECT_ROOT / args.backup_dir) if args.backup_dir else None})
    print(f"{len(tasks)} combinations{' (applying)' if args.apply else ' (report only)'}", flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as ex:
        futs = [ex.submit(one, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            r = rows[-1]
            print(f"{i}/{len(tasks)} {r['country']} {r['asset']}: {r['status']} "
                  f"{r.get('n_exposed_changed', '')}/{r.get('n_exposed', '')} exposed changed "
                  f"{r.get('error', '')}", flush=True)
    df = pd.DataFrame(rows).sort_values(["asset", "country"])
    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    name = "river_protection_applied.csv" if args.apply else "river_protection_changes.csv"
    df.to_csv(out / name, index=False)
    print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
