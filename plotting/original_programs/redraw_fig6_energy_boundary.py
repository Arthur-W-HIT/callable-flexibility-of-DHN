"""Redraw the plant-side callable heat-energy boundary figure.

The redraw mirrors the accepted heat-power boundary style so that the boundary
figure group uses one palette, legend order, marker policy, and axis-label
contract. Source data and selected action durations are unchanged.
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
    "alpha": 0.90,
}


def style_axis(ax) -> None:
    ax.grid(True, color=DEFAULT_PALETTE["grid"], linewidth=0.65, alpha=0.88)
    ax.tick_params(direction="out", length=3.0, width=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#444444")


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.020,
        0.910,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.4,
        color=DEFAULT_PALETTE["neutral_dark"],
        bbox=PANEL_BBOX,
        zorder=10,
    )


def plot_energy_family(ax, data: pd.DataFrame, *, direction: str, colors: dict[int, str]) -> None:
    direction_data = data[data["direction"] == direction].copy()
    if direction == "downward":
        y_col = "accepted_callable_energy_MWh_80MWx5_abs"
    else:
        y_col = "accepted_callable_energy_MWh_80MWx5"

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
            markersize=4.1 if duration != 345 else 5.2,
            markerfacecolor="white" if duration != 345 else colors[duration],
            markeredgecolor=colors[duration],
            markeredgewidth=0.8,
            markevery=4,
            alpha=0.96,
            zorder=3,
        )


def main() -> None:
    set_energy_style(base_size=8.0, axis_label_size=8.8, legend_size=7.2)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(SOURCE_CSV)
    data = data[data["boundary_status"].eq("usable")].copy()
    data = data[data["tau_act_min"].isin(DURATIONS_MIN)].copy()
    data["accepted_callable_energy_MWh_80MWx5_abs"] = data[
        "accepted_callable_energy_MWh_80MWx5"
    ].abs()
    copy_figure_data([SOURCE_CSV], DATA_DIR)

    duration_keys = [f"{duration}_min" for duration in DURATIONS_MIN]
    color_map_by_key = assign_series_colors(duration_keys, mode="high-impact-order")
    colors = {duration: color_map_by_key[f"{duration}_min"] for duration in DURATIONS_MIN}

    max_energy = float(
        data[
            [
                "accepted_callable_energy_MWh_80MWx5",
                "accepted_callable_energy_MWh_80MWx5_abs",
            ]
        ]
        .abs()
        .max()
        .max()
    )
    y_upper = max(380.0, ((max_energy // 25) + 2) * 25)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=cm_to_in(18.2, 7.9),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.18},
    )

    panel_specs = [
        (
            axes[0],
            "upward",
            "Upward callable energy",
            axis_label(
                "Callable heat-energy boundary",
                r"$E_{\mathrm{plant}}^{\mathrm{up}}$",
                r"MWh$_{\mathrm{th}}$",
            ),
            "(a)",
        ),
        (
            axes[1],
            "downward",
            "Downward callable energy",
            axis_label(
                "Callable heat-energy boundary",
                r"$|E_{\mathrm{plant}}^{\mathrm{down}}|$",
                r"MWh$_{\mathrm{th}}$",
            ),
            "(b)",
        ),
    ]

    for ax, direction, title, ylabel, label in panel_specs:
        plot_energy_family(ax, data, direction=direction, colors=colors)
        ax.set_title(title, fontweight="normal", fontsize=8.6, pad=3)
        ax.set_xlabel(axis_label("Ambient temperature", r"$T_{\mathrm{amb}}$", "℃"))
        ax.set_ylabel(ylabel)
        ax.set_xlim(-36.5, 16.5)
        ax.set_ylim(0, y_upper)
        style_axis(ax)
        ax.tick_params(labelleft=True)
        add_panel_label(ax, label)

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
    fig.subplots_adjust(left=0.075, right=0.985, top=0.805, bottom=0.175, wspace=0.20)

    outputs = save_figure(
        fig,
        FIG_DIR / "corrected_energy_boundary_80MWx5",
        formats=("png",),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
    )

    write_manifest(
        PACKAGE_DIR / "fig6_energy_boundary_redraw_manifest.json",
        {
            "figure": "Fig. 6 candidate",
            "figure_file_stem": "corrected_energy_boundary_80MWx5",
            "backend": "python/matplotlib",
            "source_data": [str(SOURCE_CSV)],
            "exports": [str(path.relative_to(PACKAGE_DIR)) for path in outputs],
            "palette_mode": "high-impact-order assigned by action-duration order",
            "series_order": [f"{duration} min" for duration in DURATIONS_MIN],
            "series_color_map": {f"{duration} min": colors[duration] for duration in DURATIONS_MIN},
            "semantic_accent_overrides": {},
            "axis_label_audit": (
                "Visible axes use full quantity names, mathematical symbols, and units; "
                "ambient temperature uses the Celsius symbol instead of degC."
            ),
            "information_density_decision": (
                "The figure keeps the two non-substitutable panel roles: upward and "
                "downward plant-side callable heat-energy boundaries. The two panels "
                "share one y-axis scale to support direct magnitude comparison."
            ),
            "annotation_readability_audit": (
                "Panel labels are inside the axes; no direct value labels are placed "
                "near data lines, and the compact top legend does not obscure data."
            ),
            "claim_supported": (
                "For the tested fixed-supply-flow boundary scan, plant-side callable "
                "heat-energy limits vary with ambient temperature, action direction, "
                "and action duration."
            ),
            "what_this_does_not_prove": (
                "The figure does not provide independent field validation and should "
                "not be read as a universal seasonal energy-yield estimate."
            ),
            "evidence_boundary": (
                "Model-based boundary reconstruction for five parallel 80 MW_th "
                "subnetworks under the tested baseline operating policy."
            ),
        },
    )


if __name__ == "__main__":
    main()
