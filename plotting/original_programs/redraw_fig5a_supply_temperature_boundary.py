"""Redraw the supply-temperature boundary figure.

This candidate is intended to sit before the plant-side heat-power boundary in
the manuscript figure sequence. It shows the boundary variable that drives the
heat-power response: accepted supply-temperature deviations from the baseline
and the resulting absolute accepted supply-temperature levels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = PACKAGE_DIR.parents[1]
FIG_DIR = PACKAGE_DIR / "AE_final_png_figures"
DATA_DIR = PACKAGE_DIR / "AE_final_png_figure_data"
SOURCE_CSV = (
    PROJECT_DIR
    / "AE_Rebuild"
    / "Boundary_FixedSupplyFlow_1C_CapRerun"
    / "corrected"
    / "boundary_result_table_cap_corrected.csv"
)

HELPER_CANDIDATES = [
    SCRIPT_DIR,
    SCRIPT_DIR / "helpers",
]
for candidate in HELPER_CANDIDATES:
    if (candidate / "energy_figure_style.py").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from energy_figure_style import (  # noqa: E402
    DEFAULT_PALETTE,
    assign_series_colors,
    axis_label,
    cm_to_in,
    copy_figure_data,
    save_figure,
    set_energy_style,
    write_manifest,
)


DURATIONS_MIN = [15, 30, 60, 120, 240, 345]
MARKERS = {
    15: "o",
    30: "s",
    60: "D",
    120: "^",
    240: "v",
    345: "*",
}
LINESTYLES = {
    15: "-",
    30: "--",
    60: ":",
    120: "-.",
    240: "-",
    345: (0, (4, 2, 1, 2)),
}
PANEL_BBOX = {
    "boxstyle": "round,pad=0.10",
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 0.92,
}


def style_axis(ax) -> None:
    ax.grid(True, color=DEFAULT_PALETTE["grid"], linewidth=0.62, alpha=0.88)
    ax.tick_params(direction="out", length=3.0, width=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#444444")


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.020,
        0.905,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        color=DEFAULT_PALETTE["neutral_dark"],
        bbox=PANEL_BBOX,
        zorder=10,
    )


def plot_duration_family(
    ax,
    data: pd.DataFrame,
    *,
    direction: str,
    y_col: str,
    colors: dict[int, str],
) -> None:
    direction_data = data[data["direction"] == direction].copy()
    for duration in DURATIONS_MIN:
        subset = direction_data[direction_data["tau_act_min"] == duration].sort_values("T_amb_C")
        if subset.empty:
            continue
        ax.plot(
            subset["T_amb_C"],
            subset[y_col],
            color=colors[duration],
            linestyle=LINESTYLES[duration],
            linewidth=1.45 if duration <= 120 else 1.25,
            marker=MARKERS[duration],
            markersize=4.0 if duration != 345 else 5.0,
            markerfacecolor="white" if duration != 345 else colors[duration],
            markeredgecolor=colors[duration],
            markeredgewidth=0.8,
            markevery=5,
            alpha=0.96,
            zorder=3,
        )


def add_baseline(ax, baseline: pd.DataFrame) -> None:
    ax.plot(
        baseline["T_amb_C"],
        baseline["baseline_supply_C"],
        color=DEFAULT_PALETTE["neutral_dark"],
        linestyle=(0, (4, 2)),
        linewidth=1.25,
        zorder=4,
    )
    label_x = float(baseline["T_amb_C"].max()) - 1.5
    label_y = float(
        baseline.loc[
            baseline["T_amb_C"].sub(label_x).abs().idxmin(),
            "baseline_supply_C",
        ]
    )
    ax.text(
        label_x,
        label_y + 2.8,
        r"Baseline $T_{\mathrm{s,base}}$",
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=DEFAULT_PALETTE["neutral_dark"],
        bbox=PANEL_BBOX,
        zorder=8,
    )


def rounded_limits(low: float, high: float, step: float, pad_steps: int = 1) -> tuple[float, float]:
    lower = step * int(low // step) - pad_steps * step
    upper = step * int(-(-high // step)) + pad_steps * step
    return lower, upper


def main() -> None:
    set_energy_style(base_size=8.0, axis_label_size=8.6, legend_size=7.2)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(SOURCE_CSV)
    data = data[data["boundary_status"].eq("usable")].copy()
    data = data[data["tau_act_min"].isin(DURATIONS_MIN)].copy()
    data["accepted_delta_T_supply_abs_C"] = data["accepted_delta_T_supply_C"].abs()
    data["accepted_supply_temperature_C"] = (
        data["baseline_supply_C"] + data["accepted_delta_T_supply_C"]
    )
    baseline = (
        data.groupby("T_amb_C", as_index=False)["baseline_supply_C"]
        .mean()
        .sort_values("T_amb_C")
    )
    copy_figure_data([SOURCE_CSV], DATA_DIR)

    duration_keys = [f"{duration}_min" for duration in DURATIONS_MIN]
    color_map_by_key = assign_series_colors(duration_keys, mode="high-impact-order")
    colors = {duration: color_map_by_key[f"{duration}_min"] for duration in DURATIONS_MIN}

    delta_upper = max(65.0, data["accepted_delta_T_supply_abs_C"].max() + 3.0)
    temp_low, temp_high = rounded_limits(
        min(data["accepted_supply_temperature_C"].min(), baseline["baseline_supply_C"].min()),
        max(data["accepted_supply_temperature_C"].max(), baseline["baseline_supply_C"].max()),
        5.0,
        pad_steps=1,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=cm_to_in(18.2, 11.9),
        sharex=True,
        gridspec_kw={"wspace": 0.20, "hspace": 0.34},
    )

    panel_specs = [
        (
            axes[0, 0],
            "upward",
            "accepted_delta_T_supply_abs_C",
            "Upward deviation boundary",
            axis_label(
                "Supply-temperature deviation",
                r"$\Delta T_{\mathrm{s}}^{\mathrm{up}}$",
                "℃",
            ),
            (0.0, delta_upper),
            "(a)",
            False,
        ),
        (
            axes[0, 1],
            "downward",
            "accepted_delta_T_supply_abs_C",
            "Downward deviation boundary",
            axis_label(
                "Supply-temperature deviation",
                r"$|\Delta T_{\mathrm{s}}^{\mathrm{down}}|$",
                "℃",
            ),
            (0.0, delta_upper),
            "(b)",
            False,
        ),
        (
            axes[1, 0],
            "upward",
            "accepted_supply_temperature_C",
            "Upward accepted supply temperature",
            axis_label(
                "Accepted supply temperature",
                r"$T_{\mathrm{s,act}}^{\mathrm{up}}$",
                "℃",
            ),
            (temp_low, temp_high),
            "(c)",
            True,
        ),
        (
            axes[1, 1],
            "downward",
            "accepted_supply_temperature_C",
            "Downward accepted supply temperature",
            axis_label(
                "Accepted supply temperature",
                r"$T_{\mathrm{s,act}}^{\mathrm{down}}$",
                "℃",
            ),
            (temp_low, temp_high),
            "(d)",
            True,
        ),
    ]

    for ax, direction, y_col, title, ylabel, ylim, panel_label, show_baseline in panel_specs:
        plot_duration_family(ax, data, direction=direction, y_col=y_col, colors=colors)
        if show_baseline:
            add_baseline(ax, baseline)
        ax.set_title(title, fontweight="normal", fontsize=8.5, pad=3)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-36.5, 16.5)
        ax.set_ylim(*ylim)
        style_axis(ax)
        add_panel_label(ax, panel_label)

    for ax in axes[0, :]:
        ax.tick_params(labelbottom=False)
    for ax in axes[1, :]:
        ax.set_xlabel(axis_label("Ambient temperature", r"$T_{\mathrm{amb}}$", "℃"))

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=colors[duration],
            linestyle=LINESTYLES[duration],
            marker=MARKERS[duration],
            markersize=4.2 if duration != 345 else 5.2,
            markerfacecolor="white" if duration != 345 else colors[duration],
            markeredgecolor=colors[duration],
            markeredgewidth=0.8,
            linewidth=1.35,
            label=f"{duration} min",
        )
        for duration in DURATIONS_MIN
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.55, 0.995),
        title="Action duration",
        title_fontsize=7.4,
        handlelength=1.9,
        columnspacing=1.0,
        handletextpad=0.42,
    )
    fig.subplots_adjust(left=0.078, right=0.985, top=0.865, bottom=0.105)

    outputs = save_figure(
        fig,
        FIG_DIR / "corrected_supply_temperature_boundary_80MWx5",
        formats=("png",),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
    )

    write_manifest(
        PACKAGE_DIR / "fig5a_supply_temperature_boundary_redraw_manifest.json",
        {
            "figure": "Fig. 5a candidate; insert before the heat-power boundary if adopted",
            "figure_file_stem": "corrected_supply_temperature_boundary_80MWx5",
            "backend": "python/matplotlib",
            "source_data": [str(SOURCE_CSV)],
            "exports": [str(path.relative_to(PACKAGE_DIR)) for path in outputs],
            "palette_mode": "high-impact-order assigned by action-duration order",
            "series_order": [f"{duration} min" for duration in DURATIONS_MIN],
            "series_color_map": {f"{duration} min": colors[duration] for duration in DURATIONS_MIN},
            "semantic_accent_overrides": {
                "baseline_supply_temperature": DEFAULT_PALETTE["neutral_dark"]
            },
            "panel_map": {
                "(a)": "upward accepted supply-temperature deviation from baseline",
                "(b)": "downward accepted supply-temperature deviation from baseline",
                "(c)": "upward absolute accepted supply temperature with baseline reference",
                "(d)": "downward absolute accepted supply temperature with baseline reference",
            },
            "axis_label_audit": (
                "Visible axes use complete quantity names, mathematical symbols, "
                "and Celsius symbols; no degC or source-column token is exposed."
            ),
            "information_density_decision": (
                "The figure combines relative deviation and absolute supply-temperature "
                "levels in one 2x2 evidence figure. No single monotonic curve occupies "
                "a standalone figure, and the panel roles are not interchangeable."
            ),
            "annotation_readability_audit": (
                "Panel labels and baseline labels are kept inside each axes with "
                "light text boxes; no value labels are placed on dense duration curves."
            ),
            "claim_supported": (
                "For the tested fixed-supply-flow boundary scan, accepted "
                "supply-temperature action boundaries vary with ambient temperature, "
                "action direction, and action duration, providing the upstream control "
                "boundary for the following heat-power boundary figure."
            ),
            "what_this_does_not_prove": (
                "The figure does not report a full pipe-field temperature distribution, "
                "return-temperature envelope, or independent field validation."
            ),
            "evidence_boundary": (
                "Model-based supply-temperature boundary reconstruction for five "
                "parallel 80 MW_th subnetworks under the tested baseline operating policy."
            ),
        },
    )


if __name__ == "__main__":
    main()
