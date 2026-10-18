"""Generate supplementary evidence figures for the Chinese AE revision.

The figures are post-processing diagnostics from existing CSV outputs. They do
not overwrite raw simulation or dispatch results.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR.parent
PROJECT_DIR = RUN_DIR.parents[1]
FIG_DIR = RUN_DIR / "figures"
DATA_DIR = RUN_DIR / "data"

HELPER_CANDIDATES = [
    SCRIPT_DIR,
    SCRIPT_DIR / "helpers",
]
for candidate in HELPER_CANDIDATES:
    if (candidate / "energy_figure_style.py").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from energy_figure_style import (  # noqa: E402
    DEFAULT_PALETTE,
    add_panel_label,
    axis_label,
    cm_to_in,
    save_figure,
    set_energy_style,
    write_manifest,
)


WEEKLY_DIR = PROJECT_DIR / "AE_Rebuild" / "Weekly_DecisionStep_Sensitivity" / "result_tables"
BOUNDARY_DIR = PROJECT_DIR / "AE_Rebuild" / "Boundary_FixedSupplyFlow_1C_Full"

SCHEDULE_STEP15 = WEEKLY_DIR / "schedule_step15min.csv"
BOUNDARY_CSV = BOUNDARY_DIR / "boundary_aggregated_80MWx5.csv"


def _style_axis(ax) -> None:
    ax.grid(True, which="major", axis="both")
    ax.tick_params(length=3.0, width=0.7)
    for spine in ax.spines.values():
        spine.set_color(DEFAULT_PALETTE["neutral_dark"])
        spine.set_linewidth(0.7)


def _copy_inputs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for src in [SCHEDULE_STEP15, BOUNDARY_CSV]:
        shutil.copy2(src, DATA_DIR / src.name)


def _load_step15_and_boundary() -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule = pd.read_csv(SCHEDULE_STEP15)
    boundary = pd.read_csv(BOUNDARY_CSV)
    schedule["Time_d"] = schedule["Time_h"] / 24.0
    return schedule, boundary


def _fixed_tank_state(schedule: pd.DataFrame, boundary: pd.DataFrame) -> pd.DataFrame:
    positive_cap = boundary.loc[
        boundary["direction"].eq("upward"), "accepted_callable_energy_MWh_80MWx5"
    ].max()
    negative_cap = abs(
        boundary.loc[
            boundary["direction"].eq("downward"), "accepted_callable_energy_MWh_80MWx5"
        ].min()
    )
    max_duration = boundary["tau_act_min"].max()
    positive_long_cap = boundary.loc[
        boundary["direction"].eq("upward") & boundary["tau_act_min"].eq(max_duration),
        "accepted_callable_energy_MWh_80MWx5",
    ].max()
    negative_long_cap = abs(
        boundary.loc[
            boundary["direction"].eq("downward") & boundary["tau_act_min"].eq(max_duration),
            "accepted_callable_energy_MWh_80MWx5",
        ].min()
    )

    dt_h = float(schedule["Time_h"].diff().dropna().median())
    cumulative_energy = (schedule["DeltaQ_MWth"] * dt_h).cumsum()

    def normalize(series: pd.Series, cap_up: float, cap_down: float) -> pd.Series:
        return series.where(series >= 0, series / cap_down).where(series < 0, series / cap_up)

    out = schedule[["Step", "DateTime", "Time_d", "State_x", "DeltaQ_MWth", "EffectiveDuration_min"]].copy()
    out["FixedTankState_maxScan"] = normalize(cumulative_energy, positive_cap, negative_cap)
    out["FixedTankState_longestDuration"] = normalize(
        cumulative_energy, positive_long_cap, negative_long_cap
    )
    out["CumulativeEnergy_MWh"] = cumulative_energy
    out["PositiveEnergyCapacity_maxScan_MWh"] = positive_cap
    out["NegativeEnergyCapacity_maxScan_MWh"] = negative_cap
    out["PositiveEnergyCapacity_longestDuration_MWh"] = positive_long_cap
    out["NegativeEnergyCapacity_longestDuration_MWh"] = negative_long_cap
    return out


def _occupancy_summary(state: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidates = [
        ("Occupancy state with duration memory", "State_x"),
        ("Fixed energy tank, max scanned energy", "FixedTankState_maxScan"),
        ("Fixed energy tank, longest-duration energy", "FixedTankState_longestDuration"),
    ]
    for label, col in candidates:
        series = state[col].astype(float)
        rows.append(
            {
                "Representation": label,
                "MaxState": series.max(),
                "MinState": series.min(),
                "MaxAbsState": series.abs().max(),
                "Samples_abs_state_above_0p8": int(series.abs().gt(0.8).sum()),
                "Samples_abs_state_above_0p9": int(series.abs().gt(0.9).sum()),
                "FinalState": series.iloc[-1],
            }
        )

    active = schedule.loc[schedule["ActionSign"].ne(0)].copy()
    active["BoundaryUtilizationRatio"] = active["DeltaQ_MWth"].abs() / active[
        "EffectiveBoundary_MW"
    ].replace([np.inf, -np.inf], np.nan).abs()
    active["DurationBin"] = pd.cut(
        active["EffectiveDuration_min"],
        bins=[0, 60, 180, 360],
        labels=["15-60", "75-180", "195-345"],
        include_lowest=True,
    )
    duration_summary = (
        active.groupby("DurationBin", observed=True)["BoundaryUtilizationRatio"]
        .agg(["count", "mean", "max"])
        .reset_index()
    )
    return pd.DataFrame(rows), duration_summary


def plot_occupancy_ablation() -> tuple[list[Path], dict]:
    schedule, boundary = _load_step15_and_boundary()
    state = _fixed_tank_state(schedule, boundary)
    representation_summary, duration_summary = _occupancy_summary(state, schedule)

    state.to_csv(DATA_DIR / "occupancy_state_ablation_timeseries.csv", index=False)
    representation_summary.to_csv(DATA_DIR / "occupancy_state_ablation_summary.csv", index=False)
    duration_summary.to_csv(DATA_DIR / "duration_dependent_utilization_summary.csv", index=False)

    fig = plt.figure(figsize=cm_to_in(18.2, 11.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], wspace=0.28, hspace=0.42)
    ax_state = fig.add_subplot(gs[0, :])
    ax_count = fig.add_subplot(gs[1, 0])
    ax_duration = fig.add_subplot(gs[1, 1])

    ax_state.plot(
        state["Time_d"],
        state["State_x"],
        color=DEFAULT_PALETTE["main"],
        linewidth=1.15,
        label="Occupancy state with duration memory",
    )
    ax_state.plot(
        state["Time_d"],
        state["FixedTankState_maxScan"],
        color=DEFAULT_PALETTE["contrast"],
        linewidth=1.0,
        linestyle=(0, (4, 2)),
        label="Fixed energy tank",
    )
    for y in [-0.9, 0.9]:
        ax_state.axhline(y, color=DEFAULT_PALETTE["constraint"], linewidth=0.8, linestyle=":")
    ax_state.text(
        state["Time_d"].max() * 0.99,
        0.92,
        r"$|x|=0.90$ limit",
        ha="right",
        va="bottom",
        fontsize=7.1,
        color=DEFAULT_PALETTE["constraint"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5),
    )
    ax_state.set_xlim(0, 7)
    ax_state.set_ylim(-3.15, 1.05)
    ax_state.set_xlabel(axis_label("Week time", r"$t$", "d"))
    ax_state.set_ylabel("Normalized state (-)")
    ax_state.legend(loc="lower left", frameon=False, ncol=2, fontsize=7.4)
    ax_state.set_title("State representation response to the same 15 min dispatch sequence", fontweight="normal", fontsize=8.8)
    _style_axis(ax_state)
    add_panel_label(ax_state, "(a)", fontsize=8.6)

    labels = ["Occupancy\nstate", "Fixed tank\nmax scan", "Fixed tank\nlong duration"]
    counts = representation_summary["Samples_abs_state_above_0p9"].to_numpy()
    colors = [
        DEFAULT_PALETTE["main"],
        DEFAULT_PALETTE["contrast"],
        DEFAULT_PALETTE["warning"],
    ]
    ax_count.bar(np.arange(len(labels)), counts, color=colors, width=0.62)
    ax_count.set_xticks(np.arange(len(labels)), labels)
    ax_count.set_ylabel("Samples above state limit (-)")
    ax_count.set_title(r"Apparent $|state|>0.90$ violations", fontweight="normal", fontsize=8.5)
    for i, value in enumerate(counts):
        ax_count.text(i, value + max(counts) * 0.035 + 2, f"{int(value)}", ha="center", va="bottom", fontsize=7.1)
    ax_count.set_ylim(0, max(counts) * 1.18 + 5)
    _style_axis(ax_count)
    add_panel_label(ax_count, "(b)", fontsize=8.6)

    x = np.arange(len(duration_summary))
    ax_duration.bar(
        x - 0.16,
        duration_summary["mean"],
        width=0.32,
        color=DEFAULT_PALETTE["secondary"],
        label="Mean utilization",
    )
    ax_duration.bar(
        x + 0.16,
        duration_summary["max"],
        width=0.32,
        color=DEFAULT_PALETTE["main"],
        label="Maximum utilization",
    )
    ax_duration.set_xticks(x, [f"{v} min" for v in duration_summary["DurationBin"].astype(str)])
    ax_duration.set_ylabel("Boundary utilization ratio (-)")
    ax_duration.set_xlabel(axis_label("Effective action duration", r"$k$", "min"))
    ax_duration.set_ylim(0, 0.8)
    ax_duration.legend(loc="upper left", frameon=False, fontsize=7.2)
    ax_duration.set_title("Duration dependent boundary use", fontweight="normal", fontsize=8.5)
    _style_axis(ax_duration)
    add_panel_label(ax_duration, "(c)", fontsize=8.6)

    outputs = save_figure(fig, FIG_DIR / "occupancy_state_ablation_diagnostic")
    return outputs, {
        "figure": "occupancy_state_ablation_diagnostic",
        "data": [
            "occupancy_state_ablation_timeseries.csv",
            "occupancy_state_ablation_summary.csv",
            "duration_dependent_utilization_summary.csv",
        ],
        "key_results": {
            "occupancy_max_abs": float(representation_summary.iloc[0]["MaxAbsState"]),
            "fixed_tank_max_scan_max_abs": float(representation_summary.iloc[1]["MaxAbsState"]),
            "occupancy_limit_violations": int(representation_summary.iloc[0]["Samples_abs_state_above_0p9"]),
            "fixed_tank_limit_violations": int(representation_summary.iloc[1]["Samples_abs_state_above_0p9"]),
        },
    }


def _boundary_with_cap(boundary: pd.DataFrame, cap_c: float) -> pd.DataFrame:
    out = boundary.copy()
    out["BoundarySupplyTemperature_C"] = out["baseline_supply_C"] + out["accepted_delta_T_supply_C"]
    out["EngineeringSupplyTemperatureCap_C"] = cap_c
    out["BaselineFeasibleUnderCap"] = out["baseline_supply_C"] <= cap_c
    out["AllowedDeltaTUnderCap_C"] = np.where(
        out["direction"].eq("upward"),
        np.maximum(0.0, cap_c - out["baseline_supply_C"]),
        out["accepted_delta_T_supply_abs_C"],
    )
    out["RetainedDeltaTRatio"] = np.where(
        out["accepted_delta_T_supply_abs_C"].gt(0),
        np.minimum(out["accepted_delta_T_supply_abs_C"], out["AllowedDeltaTUnderCap_C"])
        / out["accepted_delta_T_supply_abs_C"],
        1.0,
    )
    out.loc[out["direction"].eq("downward"), "RetainedDeltaTRatio"] = 1.0
    out["FilteredDeltaQ_MW_80MWx5"] = out["accepted_delta_Q_MW_80MWx5"] * out["RetainedDeltaTRatio"]
    return out


def _boundary_filter_summary(boundary: pd.DataFrame, caps: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered_tables = []
    rows = []
    for cap in caps:
        f = _boundary_with_cap(boundary, cap)
        filtered_tables.append(f)
        up = f.loc[f["direction"].eq("upward")].copy()
        affected = int(up["RetainedDeltaTRatio"].lt(0.999999).sum())
        rows.append(
            {
                "SupplyTemperatureCap_C": cap,
                "AffectedUpwardCases": affected,
                "TotalUpwardCases": int(len(up)),
                "MeanRetainedUpwardBoundaryRatio": float(up["RetainedDeltaTRatio"].mean()),
                "MinimumRetainedUpwardBoundaryRatio": float(up["RetainedDeltaTRatio"].min()),
                "BaselineInfeasibleCases": int((~up["BaselineFeasibleUnderCap"]).sum()),
            }
        )
    return pd.concat(filtered_tables, ignore_index=True), pd.DataFrame(rows)


def plot_boundary_filter() -> tuple[list[Path], dict]:
    boundary = pd.read_csv(BOUNDARY_CSV)
    caps = [125.0, 130.0]
    filtered, summary = _boundary_filter_summary(boundary, caps)
    filtered.to_csv(DATA_DIR / "supply_temperature_cap_filtered_boundary.csv", index=False)
    summary.to_csv(DATA_DIR / "supply_temperature_cap_filter_summary.csv", index=False)

    cap_main = 125.0
    f125 = filtered.loc[filtered["EngineeringSupplyTemperatureCap_C"].eq(cap_main)].copy()
    upward = f125.loc[f125["direction"].eq("upward")].copy()
    pivot = upward.pivot_table(
        index="tau_act_min",
        columns="T_amb_C",
        values="RetainedDeltaTRatio",
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)

    selected = upward.loc[upward["tau_act_min"].isin([15, 60, 180, 345])].copy()
    selected["RawBoundary_MW"] = selected["accepted_delta_Q_MW_80MWx5"]
    selected["FilteredBoundary_MW"] = selected["FilteredDeltaQ_MW_80MWx5"]

    fig = plt.figure(figsize=cm_to_in(18.2, 12.8))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.15, 0.95],
        width_ratios=[1.18, 0.82],
        wspace=0.62,
        hspace=0.72,
    )
    ax_line = fig.add_subplot(gs[0, :])
    ax_map = fig.add_subplot(gs[1, 0])
    ax_summary = fig.add_subplot(gs[1, 1])

    duration_order = [15, 60, 180, 345]
    colors = [
        DEFAULT_PALETTE["main"],
        DEFAULT_PALETTE["positive"],
        DEFAULT_PALETTE["contrast"],
        DEFAULT_PALETTE["secondary"],
    ]
    for tau, color in zip(duration_order, colors):
        s = selected.loc[selected["tau_act_min"].eq(tau)].sort_values("T_amb_C")
        ax_line.plot(
            s["T_amb_C"],
            s["RawBoundary_MW"],
            color=color,
            linewidth=1.15,
            label=f"{tau} min, network permitted",
        )
        ax_line.plot(
            s["T_amb_C"],
            s["FilteredBoundary_MW"],
            color=color,
            linewidth=1.0,
            linestyle=(0, (4, 2)),
            label=f"{tau} min, after {cap_main:.0f} \N{DEGREE CELSIUS} cap",
        )
    ax_line.set_xlabel(axis_label("Ambient temperature", r"$T_{\mathrm{amb}}$", "\N{DEGREE CELSIUS}"))
    ax_line.set_ylabel(axis_label("Upward callable heat power", r"$B^{+}_{\mathrm{plant}}$", r"MW$_{\mathrm{th}}$"))
    ax_line.set_title("Engineering supply temperature filter after network boundary reconstruction", fontweight="normal", fontsize=8.7)
    ax_line.legend(loc="upper left", frameon=False, fontsize=6.3, ncol=2, columnspacing=0.7, handlelength=2.2)
    _style_axis(ax_line)
    add_panel_label(ax_line, "(a)", fontsize=8.6)

    im = ax_map.imshow(
        pivot.to_numpy(),
        origin="lower",
        aspect="auto",
        extent=[
            float(pivot.columns.min()),
            float(pivot.columns.max()),
            float(pivot.index.min()),
            float(pivot.index.max()),
        ],
        vmin=0,
        vmax=1,
        cmap="YlGnBu",
    )
    ax_map.set_xlabel(axis_label("Ambient temperature", r"$T_{\mathrm{amb}}$", "\N{DEGREE CELSIUS}"))
    ax_map.set_ylabel(axis_label("Action duration", r"$\tau_{\mathrm{act}}$", "min"))
    ax_map.set_title(r"Retained upward boundary ratio at $T_{\mathrm{s,max}}=125$ ℃", fontweight="normal", fontsize=8.4)
    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.035)
    cbar.set_label("Retained ratio (-)", fontsize=7.4, labelpad=9)
    cbar.ax.tick_params(labelsize=7.2)
    _style_axis(ax_map)
    add_panel_label(ax_map, "(b)", fontsize=8.6)

    x = np.arange(len(summary))
    ax_summary.bar(
        x,
        summary["MeanRetainedUpwardBoundaryRatio"],
        color=[DEFAULT_PALETTE["warning"], DEFAULT_PALETTE["main"]],
        width=0.55,
    )
    ax_summary.set_xticks(x, [f"{int(v)} \N{DEGREE CELSIUS}" for v in summary["SupplyTemperatureCap_C"]])
    ax_summary.set_ylim(0, 1.05)
    ax_summary.set_ylabel("Mean retained upward boundary ratio (-)", labelpad=5)
    ax_summary.set_xlabel(axis_label("Supply temperature cap", r"$T_{\mathrm{s,max}}$", "\N{DEGREE CELSIUS}"))
    ax_summary.set_title("Filter effect across scanned cases", fontweight="normal", fontsize=8.4)
    for i, row in summary.iterrows():
        ax_summary.text(
            i,
            row["MeanRetainedUpwardBoundaryRatio"] + 0.035,
            f"{row['MeanRetainedUpwardBoundaryRatio']:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.1,
        )
    _style_axis(ax_summary)
    add_panel_label(ax_summary, "(c)", fontsize=8.6)
    fig.subplots_adjust(left=0.105, right=0.975, top=0.94, bottom=0.105)

    outputs = save_figure(fig, FIG_DIR / "supply_temperature_cap_filter")
    return outputs, {
        "figure": "supply_temperature_cap_filter",
        "data": [
            "supply_temperature_cap_filtered_boundary.csv",
            "supply_temperature_cap_filter_summary.csv",
        ],
        "key_results": summary.to_dict(orient="records"),
        "caveat": "Post processing filter on the network permitted boundary; not a calibrated source side production model.",
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _copy_inputs()
    set_energy_style(base_size=8.0, axis_label_size=8.7, legend_size=7.4)

    figures = []
    manifests = []
    for func in [plot_occupancy_ablation, plot_boundary_filter]:
        outputs, details = func()
        figures.extend(outputs)
        manifests.append(details)

    write_manifest(
        RUN_DIR / "figure_manifest.json",
        {
            "run_id": RUN_DIR.name,
            "backend": "python/matplotlib",
            "data_sources": {
                "weekly_step15_schedule": str(SCHEDULE_STEP15),
                "network_boundary": str(BOUNDARY_CSV),
            },
            "palette": "energy-results-figure high-impact default palette",
            "figures": [
                str(path.relative_to(RUN_DIR)) if path.is_relative_to(RUN_DIR) else str(path)
                for path in figures
            ],
            "figure_details": manifests,
            "qa": {
                "axis_label_audit": "visible axes use full quantity names, symbols, and units",
                "information_density": "each panel carries state comparison, constraint count, duration utilization, or cap-filter evidence",
                "claim_boundary": "post-processing diagnostics; not independent field validation or source-side calibration",
            },
        },
    )
    (RUN_DIR / "figure_manifest_readable.json").write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
