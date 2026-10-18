"""Redraw Fig. 4 as compact support-only evidence for the 6 h action block.

The manuscript-facing figure merges the previous response/recovery diagnostic
and the separate design-condition block-length estimate. It only shows evidence
that supports the selected block length. Boundary-case recovery diagnostics are
kept in the source tables and manifest, not plotted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = PACKAGE_DIR.parents[1]
FIG_DIR = PACKAGE_DIR / "AE_final_png_figures"
DATA_DIR = PACKAGE_DIR / "AE_final_png_figure_data"
SOURCE_CSV = (
    PROJECT_DIR
    / "AE_Rebuild"
    / "ActionBlock_TimeScale_Test"
    / "result_tables"
    / "action_block_case_response_summary.csv"
)
OVERALL_CSV = (
    PROJECT_DIR
    / "AE_Rebuild"
    / "ActionBlock_TimeScale_Test"
    / "result_tables"
    / "action_block_overall_summary.csv"
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


ACTION_END_H = 1.0
SELECTED_BLOCK_H = 6.0
DEGREE_CELSIUS = "\N{DEGREE CELSIUS}"
TEXT_BBOX = {
    "boxstyle": "round,pad=0.14",
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 0.88,
}
PANEL_BBOX = {
    "boxstyle": "round,pad=0.10",
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 0.92,
}


def style_axis(ax, *, xgrid: bool = True, ygrid: bool = False) -> None:
    if xgrid:
        ax.grid(True, axis="x", color=DEFAULT_PALETTE["grid"], linewidth=0.6, alpha=0.90)
    if ygrid:
        ax.grid(True, axis="y", color=DEFAULT_PALETTE["grid"], linewidth=0.6, alpha=0.85)
    ax.tick_params(direction="out", length=3.0, width=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#444444")


def format_hours(value: float) -> str:
    if abs(value - round(value)) < 1e-8:
        return f"{value:.0f} h"
    return f"{value:.2f} h"


def add_inner_panel_label(ax, label: str) -> None:
    ax.text(
        0.018,
        0.925,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.3,
        color=DEFAULT_PALETTE["neutral_dark"],
        bbox=PANEL_BBOX,
        zorder=10,
    )


def main() -> None:
    set_energy_style(base_size=8.0, axis_label_size=8.8, legend_size=7.2)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cases = pd.read_csv(SOURCE_CSV)
    overall = pd.read_csv(OVERALL_CSV).iloc[0]
    copy_figure_data([SOURCE_CSV, OVERALL_CSV], DATA_DIR)

    design = cases[cases["case_group"] == "design_condition_exact"].copy()
    direction_order = ["downward", "upward"]
    design["direction"] = pd.Categorical(design["direction"], direction_order, ordered=True)
    design = design.sort_values("direction")

    colors = assign_series_colors(
        ["90pct_response", "peak_response", "twenty_pct_recovery", "ten_pct_recovery"],
        mode="high-impact-order",
    )
    selected_color = DEFAULT_PALETTE["constraint"]

    response_90_h = float(design["max_time_to_90pct_peak_h"].max())
    peak_h = float(design["max_time_to_peak_h"].max())
    recovery_20_h = float(
        ACTION_END_H + design["max_recovery_20pct_sustained_h_after_action_end"].max()
    )
    recovery_10_h = float(
        ACTION_END_H + design["max_recovery_10pct_sustained_h_after_action_end"].max()
    )

    fig, (ax_t, ax_s) = plt.subplots(
        1,
        2,
        figsize=cm_to_in(18.2, 6.7),
        gridspec_kw={"width_ratios": [1.03, 1.0], "wspace": 0.20},
    )

    # Panel (a): down/up actions are summarized as the conservative envelope.
    row_y = {"response": 1.0, "recovery": 0.36}
    ax_t.axvspan(0.0, ACTION_END_H, color=DEFAULT_PALETTE["neutral_light"], alpha=0.32, zorder=0)
    ax_t.axvline(SELECTED_BLOCK_H, color=selected_color, linestyle="--", linewidth=1.15, zorder=2)
    for y in row_y.values():
        ax_t.hlines(y, 0.0, SELECTED_BLOCK_H, color=DEFAULT_PALETTE["grid"], linewidth=0.75, zorder=0)
    ax_t.hlines(
        row_y["response"],
        response_90_h,
        peak_h,
        color=colors["peak_response"],
        linewidth=2.2,
        alpha=0.72,
        zorder=2,
    )
    ax_t.hlines(
        row_y["recovery"],
        ACTION_END_H,
        recovery_10_h,
        color=colors["ten_pct_recovery"],
        linewidth=2.2,
        alpha=0.64,
        zorder=2,
    )
    ax_t.scatter(
        [response_90_h],
        [row_y["response"]],
        s=42,
        marker="o",
        facecolor=colors["90pct_response"],
        edgecolor="white",
        linewidth=0.55,
        zorder=4,
    )
    ax_t.scatter(
        [peak_h],
        [row_y["response"]],
        s=46,
        marker="D",
        facecolor=colors["peak_response"],
        edgecolor="white",
        linewidth=0.55,
        zorder=4,
    )
    ax_t.scatter(
        [recovery_20_h, recovery_10_h],
        [row_y["recovery"], row_y["recovery"]],
        s=[34, 44],
        marker="s",
        facecolor=[colors["twenty_pct_recovery"], colors["ten_pct_recovery"]],
        edgecolor="white",
        linewidth=0.55,
        zorder=4,
    )
    ax_t.text(
        0.50,
        1.27,
        "Action\n1 h",
        ha="center",
        va="center",
        fontsize=7.0,
        color=DEFAULT_PALETTE["neutral_dark"],
        linespacing=0.95,
        bbox=TEXT_BBOX,
        zorder=8,
    )
    ax_t.text(
        0.06,
        row_y["response"] + 0.12,
        "Response envelope",
        ha="left",
        va="center",
        fontsize=7.2,
        color=DEFAULT_PALETTE["neutral_dark"],
        bbox=TEXT_BBOX,
        zorder=8,
    )
    ax_t.text(
        0.06,
        row_y["recovery"] + 0.12,
        "Recovery envelope",
        ha="left",
        va="center",
        fontsize=7.2,
        color=DEFAULT_PALETTE["neutral_dark"],
        bbox=TEXT_BBOX,
        zorder=8,
    )
    ax_t.annotate(
        f"90% response\n{format_hours(response_90_h)}",
        xy=(response_90_h, row_y["response"]),
        xytext=(response_90_h - 0.54, row_y["response"] + 0.22),
        ha="right",
        va="bottom",
        fontsize=7.0,
        color=colors["90pct_response"],
        bbox=TEXT_BBOX,
        arrowprops={"arrowstyle": "-", "lw": 0.65, "color": colors["90pct_response"]},
        zorder=9,
    )
    ax_t.annotate(
        f"Peak\n{format_hours(peak_h)}",
        xy=(peak_h, row_y["response"]),
        xytext=(peak_h + 0.46, row_y["response"] + 0.15),
        ha="left",
        va="bottom",
        fontsize=7.0,
        color=DEFAULT_PALETTE["neutral_dark"],
        bbox=TEXT_BBOX,
        arrowprops={"arrowstyle": "-", "lw": 0.65, "color": colors["peak_response"]},
        zorder=9,
    )
    ax_t.annotate(
        f"20% residual\n{format_hours(recovery_20_h)}",
        xy=(recovery_20_h, row_y["recovery"]),
        xytext=(recovery_20_h - 0.38, row_y["recovery"] - 0.31),
        ha="right",
        va="bottom",
        fontsize=6.9,
        color=colors["twenty_pct_recovery"],
        bbox=TEXT_BBOX,
        arrowprops={"arrowstyle": "-", "lw": 0.60, "color": colors["twenty_pct_recovery"]},
        zorder=9,
    )
    ax_t.annotate(
        f"10% residual\n{format_hours(recovery_10_h)}",
        xy=(recovery_10_h, row_y["recovery"]),
        xytext=(recovery_10_h - 0.10, row_y["recovery"] + 0.34),
        ha="right",
        va="bottom",
        fontsize=7.0,
        color=colors["ten_pct_recovery"],
        bbox=TEXT_BBOX,
        arrowprops={"arrowstyle": "-", "lw": 0.65, "color": colors["ten_pct_recovery"]},
        zorder=9,
    )
    ax_t.text(
        SELECTED_BLOCK_H - 0.05,
        1.18,
        "Selected\n6 h block",
        ha="right",
        va="center",
        fontsize=7.1,
        color=selected_color,
        linespacing=0.95,
        bbox=TEXT_BBOX,
        zorder=9,
    )
    ax_t.set_xlim(-0.05, 6.35)
    ax_t.set_ylim(-0.02, 1.52)
    ax_t.set_yticks([])
    ax_t.set_xlabel(axis_label("Time from action start", r"$t$", "h"))
    ax_t.set_title(
        f"Design-condition timing envelope (-20 {DEGREE_CELSIUS})",
        fontweight="normal",
        fontsize=8.6,
        pad=3,
    )
    style_axis(ax_t)
    add_inner_panel_label(ax_t, "(a)")

    # Panel (b): aggregate sizing checks used by the manuscript text.
    summary_items = [
        (
            "90% response bound",
            float(overall["max_user_90pct_time_h_from_start_all_cases"]),
            colors["90pct_response"],
        ),
        (
            "Peak response bound",
            float(overall["max_user_peak_time_h_from_start_all_cases"]),
            colors["peak_response"],
        ),
        (
            "1 h + 20% recovery",
            float(overall["candidate_block_h_using_design_20pct_recovery"]),
            colors["twenty_pct_recovery"],
        ),
        (
            "1 h + 10% recovery",
            float(overall["candidate_block_h_using_design_10pct_recovery"]),
            colors["ten_pct_recovery"],
        ),
    ]
    y_summary = np.arange(len(summary_items), dtype=float)
    ax_s.axvline(SELECTED_BLOCK_H, color=selected_color, linestyle="--", linewidth=1.15, zorder=3)
    for idx, (label, value, color) in enumerate(summary_items):
        ax_s.hlines(idx, 0.0, value, color=color, linewidth=2.8, alpha=0.88, zorder=2)
        ax_s.scatter(
            [value],
            [idx],
            s=42,
            marker="o",
            facecolor=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=4,
        )
        ax_s.text(
            0.08,
            idx - 0.19,
            label,
            ha="left",
            va="center",
            fontsize=7.1,
            color=DEFAULT_PALETTE["neutral_dark"],
            bbox=TEXT_BBOX,
            zorder=8,
        )
        if value >= SELECTED_BLOCK_H - 0.35:
            text_x = min(value - 0.12, SELECTED_BLOCK_H - 0.18)
            ha = "right"
        else:
            text_x = value + 0.09
            ha = "left"
        ax_s.text(
            text_x,
            idx + 0.16,
            format_hours(value),
            ha=ha,
            va="center",
            fontsize=7.2,
            color=DEFAULT_PALETTE["neutral_dark"],
            bbox=TEXT_BBOX,
            zorder=8,
        )
    ax_s.text(
        SELECTED_BLOCK_H - 0.05,
        -0.62,
        "Selected\n6 h block",
        ha="right",
        va="center",
        fontsize=7.1,
        color=selected_color,
        linespacing=0.95,
        bbox=TEXT_BBOX,
        zorder=9,
    )
    ax_s.set_xlim(-0.05, 6.35)
    ax_s.set_ylim(-0.90, len(summary_items) - 0.35)
    ax_s.invert_yaxis()
    ax_s.set_yticks(y_summary)
    ax_s.set_yticklabels([])
    ax_s.tick_params(axis="y", length=0)
    ax_s.set_xlabel(axis_label("Time scale", r"$t_{\mathrm{b}}$", "h"))
    ax_s.set_title("Block-length sizing checks", fontweight="normal", fontsize=8.6, pad=3)
    style_axis(ax_s)
    add_inner_panel_label(ax_s, "(b)")

    fig.subplots_adjust(left=0.070, right=0.985, top=0.84, bottom=0.165, wspace=0.22)

    outputs = save_figure(
        fig,
        FIG_DIR / "action_block_response_and_recovery_bars",
        formats=("png",),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
    )

    write_manifest(
        PACKAGE_DIR / "fig4_action_block_redraw_manifest.json",
        {
            "figure": "Fig. 4",
            "figure_file_stem": "action_block_response_and_recovery_bars",
            "backend": "python/matplotlib",
            "source_data": [str(SOURCE_CSV), str(OVERALL_CSV)],
            "exports": [str(path.relative_to(PACKAGE_DIR)) for path in outputs],
            "merged_figure_sources": [
                "action_block_response_and_recovery_bars",
                "design_condition_block_length_estimate",
            ],
            "supersedes_as_standalone_png": "design_condition_block_length_estimate",
            "palette_mode": "high-impact-order with sparse semantic selected-block accent",
            "series_order": [
                "90pct_response",
                "peak_response",
                "twenty_pct_recovery",
                "ten_pct_recovery",
                "selected_block",
            ],
            "series_color_map": {
                **colors,
                "selected_block": selected_color,
            },
            "semantic_accent_overrides": {"selected_6h_block": "constraint"},
            "axis_label_audit": (
                "Visible quantitative axes use full quantity names, symbols, and units; "
                "temperature is rendered as the Celsius symbol and no degC token is shown."
            ),
            "information_density_decision": (
                "The previous standalone design-condition block-length estimate has been "
                "merged into Fig. 4. Panel (a) uses a compact timing strip with direct "
                "labels instead of a low-density trajectory panel and external legend. "
                "Panel (b) carries the aggregate sizing checks."
            ),
            "annotation_readability_audit": (
                "Direct labels are kept inside panel frames, near-6 h value labels are "
                "placed to the left of the selected-block guide, and white text boxes "
                "prevent grid/data lines from obscuring text."
            ),
            "excluded_or_deferred_variables": (
                "Representative boundary-case recovery failures are retained in source "
                "tables for QA/limitations, but are not plotted because they do not "
                "support the method-rationale claim of this figure."
            ),
            "claim_supported": (
                "For the tested model, the selected 6 h action block covers the response "
                "upper bound and exceeds the design-condition 1 h plus 10% residual-"
                "recovery estimate."
            ),
            "what_this_does_not_prove": (
                "The figure does not prove that 6 h is universally valid for all ambient "
                "conditions, networks, or operating policies."
            ),
            "evidence_boundary": (
                "Model-based design-condition sizing rationale; not field validation or "
                "a universal action-block rule."
            ),
        },
    )


if __name__ == "__main__":
    main()
