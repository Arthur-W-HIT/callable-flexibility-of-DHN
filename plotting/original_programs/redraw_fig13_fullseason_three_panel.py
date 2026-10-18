"""Redraw Fig. 13 as a three-panel full-season replay figure.

The former panel (d) repeated the minimum-room-temperature values already
reported in the text and Table 5. This redraw keeps the evidence-bearing
tracking panels and uses the room-temperature envelope panel to show the
16 degC severe-under-supply threshold directly.
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

FULLSEASON_DIR = (
    PROJECT_DIR
    / "AE_Rebuild"
    / "Shenyang_HeatingSeason_15min_StateLimit0p90"
    / "replay_outputs"
)
TRACKING_CSV = FULLSEASON_DIR / "tracking_samples_decision_step.csv"
ROOM_SERIES_CSV = FULLSEASON_DIR / "room_temperature_deviation_series_sampled.csv"
ROOM_SUMMARY_CSV = FULLSEASON_DIR / "room_temperature_deviation_summary.csv"

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
    copy_figure_data,
    save_figure,
    set_energy_style,
    write_manifest,
)


PANEL_BBOX = {
    "boxstyle": "round,pad=0.10",
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 0.92,
}


def case_label(case: str) -> str:
    labels = {
        "NoFlex": "NoFlex",
        "Step15min": "15 min",
    }
    return labels.get(str(case), str(case))


def style_axis(ax) -> None:
    ax.grid(True, which="major", axis="both")
    ax.tick_params(length=3.0, width=0.7)
    for spine in ax.spines.values():
        spine.set_color(DEFAULT_PALETTE["neutral_dark"])
        spine.set_linewidth(0.7)


def plot_case_lines(ax, tracking: pd.DataFrame, y_column: str, colors: dict[str, str]) -> None:
    for case in ["NoFlex", "Step15min"]:
        s = tracking[tracking["Case"].eq(case)].sort_values("time_h")
        ax.plot(
            s["time_h"] / 24.0,
            s[y_column],
            color=colors[case],
            linestyle="-" if case == "Step15min" else (0, (4, 2)),
            linewidth=1.05,
            label=case_label(case),
            zorder=3,
        )


def redraw() -> list[Path]:
    tracking = pd.read_csv(TRACKING_CSV)
    room_series = pd.read_csv(ROOM_SERIES_CSV)
    room_summary = pd.read_csv(ROOM_SUMMARY_CSV)

    colors = {
        "NoFlex": DEFAULT_PALETTE["neutral_mid"],
        "Step15min": DEFAULT_PALETTE["main"],
    }

    fig = plt.figure(figsize=cm_to_in(18.2, 10.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], wspace=0.22, hspace=0.38)
    ax_supply = fig.add_subplot(gs[0, 0])
    ax_pump = fig.add_subplot(gs[0, 1])
    ax_room = fig.add_subplot(gs[1, :])

    plot_case_lines(ax_supply, tracking, "source_T_C", colors)
    expected = tracking[tracking["Case"].eq("NoFlex")].sort_values("time_h")
    ax_supply.plot(
        expected["time_h"] / 24.0,
        expected["expected_source_T_C"],
        color=DEFAULT_PALETTE["neutral_dark"],
        linestyle=":",
        linewidth=1.0,
        label="Expected baseline",
        zorder=2,
    )
    ax_supply.set_title("Supply-temperature replay tracking", fontweight="normal", fontsize=8.5, pad=3)
    ax_supply.set_ylabel(axis_label("Supply temperature", r"$T_{\mathrm{s}}$", "\N{DEGREE CELSIUS}"))
    style_axis(ax_supply)
    add_panel_label(ax_supply, "(a)", fontsize=8.6)

    pump_tracking = tracking.copy()
    pump_tracking["pump_power_kW"] = pump_tracking["pump_power_W"] / 1000.0
    plot_case_lines(ax_pump, pump_tracking, "pump_power_kW", colors)
    ax_pump.set_title("Pump-power proxy during replay", fontweight="normal", fontsize=8.5, pad=3)
    ax_pump.set_ylabel(axis_label("Pump power", r"$P_{\mathrm{pump}}$", "kW"))
    style_axis(ax_pump)
    add_panel_label(ax_pump, "(b)", fontsize=8.6)

    x = room_series["time_h"] / 24.0
    ax_room.fill_between(
        x,
        room_series["NoFlex_room_min_C"],
        room_series["NoFlex_room_max_C"],
        color=DEFAULT_PALETTE["neutral_mid"],
        alpha=0.12,
        edgecolor="none",
        zorder=1,
    )
    ax_room.fill_between(
        x,
        room_series["Step15min_room_min_C"],
        room_series["Step15min_room_max_C"],
        color=DEFAULT_PALETTE["main"],
        alpha=0.12,
        edgecolor="none",
        zorder=1,
    )
    ax_room.plot(
        x,
        room_series["NoFlex_room_mean_C"],
        color=DEFAULT_PALETTE["neutral_mid"],
        linestyle=(0, (4, 2)),
        linewidth=1.0,
        zorder=3,
    )
    ax_room.plot(
        x,
        room_series["Step15min_room_mean_C"],
        color=DEFAULT_PALETTE["main"],
        linewidth=1.0,
        zorder=3,
    )
    ax_room.axhline(16, color=DEFAULT_PALETTE["neutral_dark"], linestyle=":", linewidth=0.9, zorder=2)
    ax_room.text(
        x.iloc[-1] * 0.99,
        16.16,
        "16 \N{DEGREE CELSIUS} severe threshold",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=DEFAULT_PALETTE["neutral_dark"],
        bbox=PANEL_BBOX,
        clip_on=True,
        zorder=4,
    )

    min_no_flex = float(room_summary.loc[room_summary["Case"].eq("NoFlex"), "room_min_C"].iloc[0])
    min_step15 = float(room_summary.loc[room_summary["Case"].eq("Step15min"), "room_min_C"].iloc[0])
    ax_room.text(
        0.01,
        0.96,
        f"Minimum room temperature: NoFlex {min_no_flex:.2f} \N{DEGREE CELSIUS}; 15 min {min_step15:.2f} \N{DEGREE CELSIUS}",
        transform=ax_room.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color=DEFAULT_PALETTE["neutral_dark"],
        bbox=PANEL_BBOX,
        zorder=4,
    )
    ax_room.set_title("Room-temperature envelope and severe under-supply threshold", fontweight="normal", fontsize=8.5, pad=3)
    ax_room.set_xlabel(axis_label("Heating-season time", r"$t$", "d"))
    ax_room.set_ylabel(axis_label("Room temperature", r"$T_{\mathrm{room}}$", "\N{DEGREE CELSIUS}"))
    ax_room.set_ylim(15.75, 24.35)
    style_axis(ax_room)
    add_panel_label(ax_room, "(c)", fontsize=8.6)

    handles = [
        Line2D([0], [0], color=DEFAULT_PALETTE["neutral_mid"], linestyle=(0, (4, 2)), linewidth=1.05, label="NoFlex"),
        Line2D([0], [0], color=DEFAULT_PALETTE["main"], linestyle="-", linewidth=1.05, label="15 min"),
        Line2D([0], [0], color=DEFAULT_PALETTE["neutral_dark"], linestyle=":", linewidth=1.0, label="Expected baseline"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.56, 0.995),
        title="Full-season replay cases",
        title_fontsize=7.4,
        columnspacing=1.1,
        handlelength=2.0,
        handletextpad=0.40,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.105)
    outputs = save_figure(
        fig,
        FIG_DIR / "fullseason_replay_and_room_deviation",
        formats=("png",),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
    )

    copied_data = copy_figure_data([TRACKING_CSV, ROOM_SERIES_CSV, ROOM_SUMMARY_CSV], DATA_DIR)
    write_manifest(
        PACKAGE_DIR / "fig13_fullseason_three_panel_redraw_manifest.json",
        {
            "figure": "Fig. 13",
            "claim": "The 15 min replay is executable in the detailed model while remaining above the severe under-supply threshold.",
            "panel_map": {
                "a": "supply-temperature replay tracking",
                "b": "pump-power proxy during replay",
                "c": "room-temperature envelope with 16 degC severe-under-supply threshold",
            },
            "deleted_panel": "minimum-room-temperature bar chart, because the same values are reported in text and Table 5",
            "figure_files": [str(p.relative_to(PACKAGE_DIR)) for p in outputs],
            "figure_data": [str(p.relative_to(PACKAGE_DIR)) for p in copied_data],
            "axis_label_audit": "complete: visible axes use full quantity names, symbols, and publication units",
            "information_density_audit": "complete: panel d was removed as redundant; remaining panels each carry a distinct evidence role",
            "annotation_readability_audit": "checked by exported PNG visual review",
            "verification_status": "pending_pdf_layout_check",
        },
    )
    return outputs


def main() -> None:
    set_energy_style(base_size=8.0, axis_label_size=8.6, legend_size=7.2)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    outputs = redraw()
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
