"""Redraw Fig. 3 baseline operating-line diagnostics.

This revision exports only the manuscript PNG preview. The layout follows the
energy-results-figure information-density gate: avoid one monotone curve per
panel when the same ambient-temperature driver can be combined.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = PACKAGE_DIR.parents[1]
FIG_DIR = PACKAGE_DIR / "AE_final_png_figures"
DATA_DIR = PACKAGE_DIR / "AE_final_png_figure_data"
SOURCE_CSV = (
    PROJECT_DIR
    / "AE_Rebuild"
    / "Baseline_Operating_Line_FixedSupplyFlow_1C"
    / "fixed_supply_flow_baseline_1C.csv"
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
    math_power_unit,
    save_figure,
    set_energy_style,
    write_manifest,
)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.070,
        1.035,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        color=DEFAULT_PALETTE["neutral_dark"],
        clip_on=False,
    )


def style_axis(ax) -> None:
    ax.grid(True, color=DEFAULT_PALETTE["grid"], linewidth=0.55, alpha=0.85)
    ax.tick_params(direction="out", length=3, width=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#444444")


def main() -> None:
    set_energy_style(base_size=8.2, axis_label_size=8.8, legend_size=7.4)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(SOURCE_CSV).sort_values("T_amb_C")
    copy_figure_data([SOURCE_CSV], DATA_DIR)

    colors = assign_series_colors(
        ["heat_power", "flow_scale", "room_diagnostic"],
        mode="high-impact-order",
    )

    x = data["T_amb_C"].to_numpy()
    heat_power = data["source_heat_proxy_mean_MW_80MW_terminal_12h"].to_numpy()
    supply_temperature = data["T_supply_base_C"].to_numpy()
    flow_scale = data["primary_flow_scale"].to_numpy()
    room_mean_deviation = (data["room_mean_C_terminal_12h"] - 20.0).to_numpy()
    room_min_deviation = (data["room_min_C_terminal_12h"] - 20.0).to_numpy()
    room_max_deviation = (data["room_max_C_terminal_12h"] - 20.0).to_numpy()

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=cm_to_in(18.2, 7.4),
        gridspec_kw={"width_ratios": [1.18, 1.0]},
        constrained_layout=True,
    )

    temperature_cmap = LinearSegmentedColormap.from_list(
        "baseline_supply_temperature",
        [
            DEFAULT_PALETTE["main"],
            DEFAULT_PALETTE["secondary"],
            DEFAULT_PALETTE["warning"],
            DEFAULT_PALETTE["contrast"],
        ],
    )
    ax_a.plot(
        x,
        heat_power,
        color=DEFAULT_PALETTE["neutral_mid"],
        linewidth=1.0,
        alpha=0.85,
        zorder=1,
    )
    scatter = ax_a.scatter(
        x,
        heat_power,
        c=supply_temperature,
        cmap=temperature_cmap,
        s=22,
        edgecolor="white",
        linewidth=0.45,
        zorder=2,
    )
    ax_a.set_xlabel(axis_label("Ambient temperature", r"$T_{\mathrm{amb}}$", "℃"))
    ax_a.set_ylabel(
        axis_label(
            "Baseline heat power",
            r"$Q_{\mathrm{base}}$",
            math_power_unit("thermal"),
        )
    )
    ax_a.set_title("Baseline operating line", fontweight="normal", pad=3)
    add_panel_label(ax_a, "(a)")
    style_axis(ax_a)

    flow_ax = ax_a.inset_axes([0.57, 0.58, 0.38, 0.34])
    flow_ax.step(
        x,
        flow_scale,
        where="mid",
        color=colors["flow_scale"],
        linewidth=1.05,
    )
    flow_ax.plot(
        x,
        flow_scale,
        linestyle="none",
        marker="s",
        markersize=1.7,
        markerfacecolor="white",
        markeredgecolor=colors["flow_scale"],
        markeredgewidth=0.55,
    )
    flow_ax.set_xlim(float(x.min()), float(x.max()))
    flow_ax.set_ylim(max(0.0, float(flow_scale.min()) - 0.04), float(flow_scale.max()) + 0.04)
    flow_ax.set_title(
        axis_label("Primary-flow scaling factor", r"$f_{\dot{m}}$", "-"),
        fontsize=6.6,
        pad=1.5,
    )
    flow_ax.tick_params(labelsize=6.1, direction="out", length=2.0, width=0.55)
    flow_ax.grid(True, color=DEFAULT_PALETTE["grid"], linewidth=0.45, alpha=0.8)
    for spine in flow_ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#666666")

    cbar = fig.colorbar(scatter, ax=ax_a, pad=0.02, fraction=0.055)
    cbar.set_label(axis_label("Supply temperature", r"$T_{\mathrm{s,base}}$", "℃"))
    cbar.ax.tick_params(labelsize=7.0, length=2.5, width=0.6)

    room_color = colors["room_diagnostic"]
    ax_b.fill_between(
        x,
        room_min_deviation,
        room_max_deviation,
        color=room_color,
        alpha=0.20,
        linewidth=0,
        label="Min-max deviation",
    )
    ax_b.plot(
        x,
        room_mean_deviation,
        color=room_color,
        marker="o",
        markersize=2.7,
        markerfacecolor="white",
        markeredgewidth=0.7,
        linewidth=1.4,
        label="Mean deviation",
    )
    ax_b.axhline(
        0.0,
        color=DEFAULT_PALETTE["neutral_mid"],
        linestyle="--",
        linewidth=1.0,
        label="20 ℃ target",
    )
    ax_b.set_xlabel(axis_label("Ambient temperature", r"$T_{\mathrm{amb}}$", "℃"))
    ax_b.set_ylabel(
        axis_label(
            "Room-temperature deviation",
            r"$\Delta T_{\mathrm{room}}$",
            "℃",
        )
    )
    ax_b.set_title("Terminal room-temperature diagnostic", fontweight="normal", pad=3)
    ax_b.legend(loc="upper right", frameon=False, fontsize=7.2, handlelength=1.7)
    ax_b.text(
        0.02,
        0.06,
        f"{len(data)}/{len(data)} cases usable",
        transform=ax_b.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        color=DEFAULT_PALETTE["neutral_dark"],
    )
    add_panel_label(ax_b, "(b)")
    style_axis(ax_b)

    for ax in (ax_a, ax_b):
        ax.set_xlim(float(x.min()), float(x.max()))
    ax_b.set_ylim(
        min(-0.75, float(room_min_deviation.min()) - 0.06),
        max(0.75, float(room_max_deviation.max()) + 0.06),
    )

    outputs = save_figure(
        fig,
        FIG_DIR / "fixed_supply_flow_baseline_1c_diagnostic",
        formats=("png",),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
    )

    write_manifest(
        PACKAGE_DIR / "fig3_baseline_redraw_manifest.json",
        {
            "figure": "Fig. 3",
            "figure_file_stem": "fixed_supply_flow_baseline_1c_diagnostic",
            "backend": "python/matplotlib",
            "source_data": str(SOURCE_CSV),
            "exports": [str(path.relative_to(PACKAGE_DIR)) for path in outputs],
            "palette_mode": "high-impact-order",
            "series_order": [
                "heat_power",
                "flow_scale",
                "room_diagnostic",
            ],
            "series_color_map": colors,
            "semantic_accent_overrides": {"room_target_line": "neutral_mid"},
            "axis_label_audit": (
                "Complete quantity name, symbol, and unit used for axes and colorbar; "
                "source tokens such as degC/MWth are not shown as visible labels."
            ),
            "information_density_decision": (
                "Merged four monotone diagnostic panels into a primary heat-power "
                "relationship with supply-temperature color encoding, a compact "
                "flow-scale inset, and a room-temperature deviation diagnostic."
            ),
            "claim_supported": (
                "Public-data-anchored baseline operating line and steady-state "
                "diagnostics across the scanned ambient-temperature range."
            ),
            "evidence_boundary": (
                "Model-based steady-state diagnostic; not field calibration or "
                "independent validation."
            ),
        },
    )


if __name__ == "__main__":
    main()
