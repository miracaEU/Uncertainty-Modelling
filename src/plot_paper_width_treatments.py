"""fig01 (mean version) with two or three protection treatments per hazard -> paper/.

Each infrastructure row splits into one sub-row per treatment: one dot per country,
coloured by treatment, and that treatment's median across countries as an outlined
tick in the same colour. Width = (p95 - p5) / mean of EAD, from the quantile cache
paper/data/ead_quantiles.parquet (the fig01 source CSV has no November treatments).

    two     floods: as recorded vs recorded x 0.5-2
            wind:   fixed RP 50 vs sampled RP 25-200 (wind has no x 0.5-2 run)
    three   floods: + sampled RP 5-200
            wind:   + no design standard

Earthquake has no protection, so its panel has one sub-row. A treatment not run yet
for an asset (river x 0.5-2 outside roads, rail and telecom; wind with no design
standard outside rail and telecom) is marked "not run yet" in its sub-row.

    python -m src.plot_paper_width_treatments
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .plot_paper_infra import (ASSETS, DATA, FULL_WIDTH, GRID, INK, INK_2, MEAN_WIDTH_MAX, MUTED, OUT, asset_axis,
                               save, set_style, y_positions)
from .plot_paper_inputs import TREATMENT_COLOR_ALT, TREATMENTS
from .plot_paper_ranges import QUANTILES

TREATMENT_OF = {s: t for t, (_, ss) in TREATMENTS.items() for s in ss}
SETS = {
    "two": {
        "River flood": ["flood_noprot_ds", "flood_baseline_ds_sc0p5to2p0"],
        "Coastal flood": ["coastal_noprot_ds_slr", "coastal_baseline_ds_slr_sc0p5to2p0"],
        "Earthquake": ["earthquake"],
        "Windstorm": ["windstorm", "windstorm_absprot"],
    },
    "three": {
        "River flood": ["flood_noprot_ds", "flood_baseline_ds_sc0p5to2p0", "flood_absprot_ds"],
        "Coastal flood": ["coastal_noprot_ds_slr", "coastal_baseline_ds_slr_sc0p5to2p0", "coastal_absprot_ds_slr"],
        "Earthquake": ["earthquake"],
        "Windstorm": ["windstorm", "windstorm_absprot", "windstorm_unprotected"],
    },
}
LABEL = {
    "flood_noprot_ds": "As recorded (floods)", "flood_baseline_ds_sc0p5to2p0": "Recorded × 0.5–2 (floods)",
    "flood_absprot_ds": "Sampled, RP 5–200 (floods)", "windstorm": "Fixed, RP 50 (wind)",
    "windstorm_absprot": "Sampled, RP 25–200 (wind)", "windstorm_unprotected": "No design standard (wind)",
    "earthquake": "Earthquake",
}
LABEL |= {"coastal_" + k.split("_", 1)[1].replace("_ds", "_ds_slr", 1): v for k, v in LABEL.items()
          if k.startswith("flood_")}


def load() -> pd.DataFrame:
    q = pd.read_parquet(QUANTILES, columns=["country", "asset", "scenario", "mean", "q0050", "q0950"])
    q = q[q["mean"] > 0].copy()
    q["w90_mean"] = (q.q0950 - q.q0050) / q["mean"]
    return q


def fig(q: pd.DataFrame, name: str, scenarios: dict[str, list[str]]) -> pd.DataFrame:
    k = max(len(s) for s in scenarios.values())
    span = 0.62 if k == 2 else 0.7                   # vertical room per asset used by the sub-rows
    step = span / k
    positions = y_positions()
    fig_, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 3.5 if k == 2 else 4.3), sharey=True,
                              gridspec_kw={"wspace": 0.12})
    rng = np.random.default_rng(20261109)
    x_max = MEAN_WIDTH_MAX
    rows, n_off, waiting = [], 0, False
    planned_assets = {h: set(q[q.scenario.isin(s)].asset) for h, s in scenarios.items()}
    for ax, (hazard, scens) in zip(axes, scenarios.items()):
        offsets = (np.arange(len(scens)) - (len(scens) - 1) / 2) * step
        for asset, y0 in positions.items():
            if asset not in planned_assets[hazard]:
                continue
            for scen, dy in zip(scens, offsets):
                y = y0 + dy
                color = TREATMENT_COLOR_ALT[TREATMENT_OF[scen]]
                values = q.loc[(q.scenario == scen) & (q.asset == asset), "w90_mean"].to_numpy()
                if not values.size:
                    waiting = True
                    ax.text(x_max - 0.15, y, "not run yet", fontsize=4.8, color=MUTED, va="center", ha="right",
                            style="italic")
                    continue
                on = values <= x_max
                jitter = y + rng.uniform(-0.38, 0.38, values.size) * step
                ax.scatter(values[on], jitter[on], s=5 if k == 3 else 6, color=color, alpha=0.5,
                           linewidths=0, zorder=2)
                ax.scatter(np.full((~on).sum(), x_max * 0.97), jitter[~on], s=9, marker=">", color=color,
                           linewidths=0, zorder=2, clip_on=False)
                n_off += int((~on).sum())
                med = np.median(values)
                ax.plot([med] * 2, [y - 0.45 * step, y + 0.45 * step], color=color, lw=1.8,
                        solid_capstyle="butt", zorder=4,
                        path_effects=[pe.Stroke(linewidth=3.0, foreground=INK), pe.Normal()])
                rows.append({"hazard": hazard, "asset": asset, "scenario": scen, "treatment": LABEL[scen],
                             "n_countries": values.size, "median": med, "min": values.min(),
                             "max": values.max(), "off_scale": int((~on).sum())})
        ax.set_xlim(-0.3, x_max)
        ax.set_xticks([0, 5, 10])
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(hazard)
        asset_axis(ax, positions, labels=ax is axes[0])
    fig_.supxlabel("(p95 − p5) / mean of EAD", fontsize=8, color=INK_2, y=-0.03)
    order = [scenarios[h][i] for h in ("River flood", "Windstorm") for i in range(len(scenarios[h]))]
    order = sorted(order, key=lambda s: TREATMENT_OF[s] == "No protection") + scenarios["Earthquake"]
    used = {TREATMENT_OF[s]: [] for s in order}
    for s in order:
        used[TREATMENT_OF[s]].append(LABEL[s])
    handles = [Line2D([], [], ls="", marker="o", ms=4, color=TREATMENT_COLOR_ALT[t], label=" / ".join(names))
               for t, names in used.items()]
    handles.append(Line2D([], [], color=MUTED, lw=1.8, label="Median across countries",
                          path_effects=[pe.Stroke(linewidth=3.0, foreground=INK), pe.Normal()]))
    if n_off:
        handles.append(Line2D([], [], ls="", marker=">", ms=4, color=MUTED, label="Off scale"))
    fig_.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.93), ncol=2,
                handletextpad=0.3, columnspacing=1.4)
    if waiting:
        fig_.text(0.5, -0.09 if k == 2 else -0.06, "Not run yet: river × 0.5–2 outside roads, rail and telecom"
                  + ("; wind with no design standard outside rail and telecom" if k == 3 else "")
                  + " (waiting for the DamageScanner polygon fix).", ha="center", fontsize=6, color=INK_2)
    print(f"  {name}: {n_off} country values beyond the axis")
    save(fig_, name)
    out = pd.DataFrame(rows)
    out.to_csv(DATA / f"{name}.csv", index=False)
    return out


def main() -> None:
    set_style()
    q = load()
    for tag, scenarios in SETS.items():
        fig(q, f"fig01_uncertainty_width_mean_{tag}_treatments", scenarios)


if __name__ == "__main__":
    main()
