"""Saltelli design layout: with or without the second-order (BA) block.

A Saltelli sample of base size N and k factors has N repeating blocks of
  (2k+2) rows  A, AB_1..AB_k, BA_1..BA_k, B   when second-order indices are wanted
  (k+2) rows   A, AB_1..AB_k, B               when they are not.
S1 and ST come from A, AB_i and B alone, with the same estimators in both
layouts; the BA rows only serve S2. Dropping them cuts the model runs by
k/(2k+2), roughly 40-45%, without changing how S1/ST are estimated.

New samples follow `sobol.second_order` in the config (default true, the
original design). Existing archives are always read with the layout they were
written in, inferred from their row count, so both kinds can sit side by side.
"""

from __future__ import annotations

import re
from pathlib import Path

_ARCHIVE_N = re.compile(r"_sobol_n(\d+)_")


def second_order_enabled(cfg: dict) -> bool:
    """Whether NEW Sobol samples include the second-order (BA) rows."""
    return bool((cfg.get("sobol") or {}).get("second_order", True))


def runs_per_base(k: int, second_order: bool) -> int:
    return 2 * k + 2 if second_order else k + 2


def archive_n(path: str | Path) -> int | None:
    """Base N from an experiments_..._sobol_n<N>_<stamp>.tar.gz name, if present."""
    m = _ARCHIVE_N.search(Path(path).name)
    return int(m.group(1)) if m else None


def _pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def sobol_layout(n_runs: int, k: int, n: int | None = None) -> tuple[int, bool]:
    """(base N, second_order) of a Saltelli sample with n_runs rows and k factors.

    With N known (from the archive name) the block size decides. Without it,
    a power-of-two N picks the layout: N(2k+2) = N'(k+2) would need
    N'/N = (2k+2)/(k+2), which lies strictly between 1 and 2, so at most one
    layout gives a power-of-two N. Raises ValueError rather than guess.
    """
    if n is not None:
        for so in (True, False):
            if n_runs == n * runs_per_base(k, so):
                return n, so
        raise ValueError(f"{n_runs} runs do not fit N={n} with k={k} factors "
                         f"(expected {n * (2 * k + 2)} or {n * (k + 2)})")
    fits = [(n_runs // runs_per_base(k, so), so) for so in (True, False)
            if n_runs % runs_per_base(k, so) == 0]
    pow2 = [f for f in fits if _pow2(f[0])]
    if len(pow2) == 1:
        return pow2[0]
    if len(fits) == 1:
        return fits[0]
    raise ValueError(f"cannot tell the Saltelli layout of {n_runs} runs with k={k} factors")
