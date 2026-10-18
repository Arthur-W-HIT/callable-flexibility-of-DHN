"""Redraw and consolidate the remaining manuscript result figures.

The script keeps the accepted boundary and baseline figures untouched. It
redraws the remaining evidence blocks that were still based on the original
MATLAB batch output:

1. duration-dependent boundary law;
2. weekly dispatch plus occupancy-state evidence;
3. pump-cost and pump-price sensitivity evidence;
4. full-season replay plus room-temperature deviation evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = PACKAGE_DIR.parents[1]
FIG_DIR = PACKAGE_DIR / "AE_final_png_figures"
DATA_DIR = PACKAGE_DIR / "AE_final_png_figure_data"

BOUNDARY_CSV = (
    PROJECT_DIR
    / "AE_Rebuild"
    / "Boundary_FixedSupplyFlow_1C_CapRerun"
    / "corrected"
    / "boundary_result_table_cap_corrected.csv"
)
WEEKLY_DIR = PROJECT_DIR / "AE_Rebuild" / "Weekly_DecisionStep_Sensitivity" / "result_tables"
WEEKLY_SUMMARY_CSV = WEEKLY_DIR / "weekly_decision_step_dispatch_summary.csv"
WEEKLY_OCCUPANCY_CSV = WEEKLY_DIR / "occupancy_state_trajectory_by_decision_step.csv"
PUMP_DIR = PROJECT_DIR / "AE_Rebuild" / "Reviewer_Closure_WeeklyStepPump6h" / "result_tables"
PUMP_CASE_CSV = PUMP_DIR / "corrected_pump_price_case_summary.csv"
PUMP_SENSITIVITY_CSV = PUMP_DIR / "corrected_pump_price_sensitivity_matrix.csv"
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
    assign_series_colors,
    axis_label,
    cm_to_in,
    copy_figure_data,
    save_figure,
    set_energy_style,
    write_manifest,
)


DURATIONS_MIN = [15, 30, 60, 120, 240, 345]
AMBIENT_POINTS_C = [-35, -25, -15, -5, 5, 15]
STEP_CASES = ["Step15min", "Step30min", "Step45min", "Step60min", "Step120min"]
REPRESENTATIVE_CASES = ["Step15min", "Step60min", "Step120min"]
MARKERS = {
    "Step15min": "o",
    "Step30min": "s",
    "Step45min": "D",
    "Step60min": "^",
    "Step120min": "v",
}
LINESTYLES = {
    "Step15min": "-",
    "Step30min": "--",
    "Step45min": ":",
    "Step60min": "-.",
    "Step120min": "-",
}
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
        "Step30min": "30 min",
        "Step45min": "45 min",
        "Step60min": "60 min",
        "Step120min": "120 min",
    }
    return labels.get(str(case), str(case))


def step_colors() -> dict[str, str]:
    assigned = assign_series_colors([case_label(c) for c in STEP_CASES], mode="high-impact-order")
    return {case: assigned[case_label(case)] for case in STEP_CASES}


def ambient_colors() -> dict[int, str]:
    keys = [f"{temp:+d} ℃" for temp in AMBIENT_POINTS_C]
    assigned = assign_series_colors(keys, mode="high-impact-order")
    return {temp: assigned[f"{temp:+d} ℃"] for temp in AMBIENT_POINTS_C}


def style_axis(ax) -> None:
    ax.grid(True, color=DEFAULT_PALETTE["grid"], linewidth=0.62, alpha=0.88)
    ax.tick_params(direction="out", length=3.0, width=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#444444")


def add_panel_label(ax, label: str, *, y: float = 0.905) -> None:
    ax.text(
        0.020,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        color=DEFAULT_PALETTE["neutral_dark"],
        bbox=PANEL_BBOX,
        zorder=10,
    )


def set_symmetric_ylim(ax, values: pd.Series | np.ndarray, *, min_span: float = 1.0) -> None:
    vmax = float(np.nanmax(np.abs(values)))
    upper = max(min_span, np.ceil(vmax / min_span) * min_span)
    ax.set_ylim(-upper, upper)


def plot_signed_duration_panel(
    ax,
    data: pd.DataFrame,
    *,
    value_col: str,
    colors: dict[int, str],
) -> None:
    for temp in AMBIENT_POINTS_C:
        subset = data[data["T_amb_C"].eq(temp)]
        if subset.empty:
            continue
        for direction, sign, linestyle in [("upward", 1.0, "-"), ("downward", -1.0, "--")]:
            s = subset[subset["direction"].eq(direction)].sort_values("tau_act_min")
            if s.empty:
                continue
            ax.plot(
                s["tau_act_min"],
                sign * s[value_col].abs(),
                color=colors[temp],
                linestyle=linestyle,
                linewidth=1.30,
                marker="o",
                markersize=3.5,
                markerfacecolor="white",
                markeredgecolor=colors[temp],
                markeredgewidth=0.75,
                alpha=0.96,
                zorder=3,
            )
    ax.axhline(0, color=DEFAULT_PALETTE["neutral_mid"], linestyle=":", linewidth=0.95, zorder=2)
    style_axis(ax)
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
    ax.text(
        0.965,
        0.90,
        "Upward",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=7.2,
        color=DEFAULT_PALETTE["neutral_dark"],
        bbox=PANEL_BBOX,
    )
    ax.text(
        0.965,
        0.10,
        "Downward",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=7.2,
        color=DEFAULT_PALETTE["neutral_dark"],
        bbox=PANEL_BBOX,
    )


def redraw_duration_boundary() -> list[Path]:
    data = pd.read_csv(BOUNDARY_CSV)
    data = data[data["boundary_status"].eq("usable")].copy()
    data = data[data["tau_act_min"].isin(DURATIONS_MIN)].copy()
    colors = ambient_colors()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=cm_to_in(18.2, 8.4),
        sharex=True,
        gridspec_kw={"wspace": 0.20},
    )

    plot_signed_duration_panel(
        axes[0],
        data,
        value_col="accepted_delta_Q_MW_80MWx5",
        colors=colors,
    )
    axes[0].set_title("Duration law for heat-power boundary", fontweight="normal", fontsize=8.5, pad=3)
    axes[0].set_xlabel(axis_label("Action duration", r"$\tau_{\mathrm{act}}$", "min"))
    axes[0].set_ylabel(
        axis_label(
            "Signed callable heat-power boundary",
            r"$B_{\mathrm{plant}}$",
            r"MW$_{\mathrm{th}}$",
        )
    )
    set_symmetric_ylim(axes[0], data["accepted_delta_Q_MW_80MWx5"], min_span=50.0)
    add_panel_label(axes[0], "(a)")

    plot_signed_duration_panel(
        axes[1],
        data,
        value_col="accepted_callable_energy_MWh_80MWx5",
        colors=colors,
    )
    axes[1].set_title("Duration law for heat-energy boundary", fontweight="normal", fontsize=8.5, pad=3)
    axes[1].set_xlabel(axis_label("Action duration", r"$\tau_{\mathrm{act}}$", "min"))
    axes[1].set_ylabel(
        axis_label(
            "Signed callable heat-energy boundary",
            r"$E_{\mathrm{plant}}$",
            r"MWh$_{\mathrm{th}}$",
        )
    )
    set_symmetric_ylim(axes[1], data["accepted_callable_energy_MWh_80MWx5"], min_span=100.0)
    add_panel_label(axes[1], "(b)")

    ambient_handles = [
        Line2D(
            [0],
            [0],
            color=colors[temp],
            linestyle="-",
            marker="o",
            markersize=3.6,
            markerfacecolor="white",
            markeredgecolor=colors[temp],
            linewidth=1.25,
            label=f"{temp:+d} ℃",
        )
        for temp in AMBIENT_POINTS_C
    ]
    direction_handles = [
        Line2D([0], [0], color=DEFAULT_PALETTE["neutral_dark"], linestyle="-", linewidth=1.25, label="Upward"),
        Line2D([0], [0], color=DEFAULT_PALETTE["neutral_dark"], linestyle="--", linewidth=1.25, label="Downward"),
    ]
    fig.legend(
        handles=ambient_handles + direction_handles,
        loc="upper center",
        ncol=8,
        frameon=False,
        bbox_to_anchor=(0.55, 0.995),
        title="Ambient temperature and action direction",
        title_fontsize=7.4,
        columnspacing=0.85,
        handlelength=1.7,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.78, bottom=0.17)
    return save_figure(
        fig,
        FIG_DIR / "curve_signed_boundary_vs_duration_selected_ambient",
        formats=("png",),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
    )


def load_weekly_schedules() -> dict[str, pd.DataFrame]:
    files = {
        "NoFlex": "schedule_noflex.csv",
        "Step15min": "schedule_step15min.csv",
        "Step30min": "schedule_step30min.csv",
        "Step45min": "schedule_step45min.csv",
        "Step60min": "schedule_step60min.csv",
        "Step120min": "schedule_step120min.csv",
    }
    return {case: pd.read_csv(WEEKLY_DIR / filename) for case, filename in files.items()}


def redraw_weekly_dispatch_and_occupancy() -> list[Path]:
    schedules = load_weekly_schedules()
    occupancy = pd.read_csv(WEEKLY_OCCUPANCY_CSV)
    summary = pd.read_csv(WEEKLY_SUMMARY_CSV)
    colors = step_colors()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=cm_to_in(18.2, 12.2),
        sharex=False,
        gridspec_kw={"wspace": 0.20, "hspace": 0.36},
    )

    ax = axes[0, 0]
    ref = schedules["NoFlex"]
    ax.plot(
        ref["Time_h"],
        ref["Qref_MWth"],
        color=DEFAULT_PALETTE["neutral_dark"],
        linestyle=(0, (4, 2)),
        linewidth=1.15,
        label="Reference",
        zorder=2,
    )
    for case in REPRESENTATIVE_CASES:
        s = schedules[case]
        ax.plot(
            s["Time_h"],
            s["Qh_MWth"],
            color=colors[case],
            linestyle=LINESTYLES[case],
            linewidth=1.10,
            label=case_label(case),
            zorder=3,
        )
    ax.set_title("Heat dispatch trajectories", fontweight="normal", fontsize=8.5, pad=3)
    ax.set_ylabel(axis_label("Heat power", r"$Q_{\mathrm{h}}$", r"MW$_{\mathrm{th}}$"))
    ax.set_xlim(0, 168)
    style_axis(ax)
    add_panel_label(ax, "(a)")

    ax = axes[0, 1]
    for case in REPRESENTATIVE_CASES:
        s = schedules[case]
        ax.plot(
            s["Time_h"],
            s["DeltaQ_MWth"],
            color=colors[case],
            linestyle=LINESTYLES[case],
            linewidth=1.10,
            label=case_label(case),
            zorder=3,
        )
    ax.axhline(0, color=DEFAULT_PALETTE["neutral_mid"], linestyle=":", linewidth=0.9, zorder=2)
    ax.set_title("Callable heat adjustment", fontweight="normal", fontsize=8.5, pad=3)
    ax.set_ylabel(axis_label("Heat-power adjustment", r"$\Delta Q_{\mathrm{h}}$", r"MW$_{\mathrm{th}}$"))
    ax.set_xlim(0, 168)
    style_axis(ax)
    add_panel_label(ax, "(b)")

    ax = axes[1, 0]
    for case in REPRESENTATIVE_CASES:
        s = occupancy[occupancy["Case"].eq(case)].sort_values("Time_h")
        ax.plot(
            s["Time_h"],
            s["State_x"],
            color=colors[case],
            linestyle=LINESTYLES[case],
            linewidth=1.10,
            label=case_label(case),
            zorder=3,
        )
    for value in [-0.9, 0.0, 0.9]:
        ax.axhline(
            value,
            color=DEFAULT_PALETTE["neutral_mid"],
            linestyle="--" if abs(value) > 0 else ":",
            linewidth=0.9,
            zorder=2,
        )
    ax.set_title("Flexibility occupancy state", fontweight="normal", fontsize=8.5, pad=3)
    ax.set_xlabel(axis_label("Time in selected week", r"$t$", "h"))
    ax.set_ylabel(axis_label("Occupancy state", r"$x$", "-"))
    ax.set_xlim(0, 168)
    ax.set_ylim(-1.02, 1.02)
    style_axis(ax)
    add_panel_label(ax, "(c)")

    ax = axes[1, 1]
    step_summary = summary[summary["Case"].isin(STEP_CASES)].copy()
    step_summary["label"] = step_summary["Case"].map(case_label)
    x = np.arange(len(step_summary))
    ax.bar(
        x,
        step_summary["UpwardActionCount"],
        width=0.62,
        color=DEFAULT_PALETTE["main"],
        edgecolor="none",
        label="Upward",
        zorder=3,
    )
    ax.bar(
        x,
        -step_summary["DownwardActionCount"],
        width=0.62,
        color=DEFAULT_PALETTE["contrast"],
        edgecolor="none",
        label="Downward",
        zorder=3,
    )
    ax.axhline(0, color=DEFAULT_PALETTE["neutral_mid"], linewidth=0.8, zorder=2)
    ax.set_title("Action composition by decision step", fontweight="normal", fontsize=8.5, pad=3)
    ax.set_xlabel(axis_label("Decision step", r"$\Delta t_{\mathrm{d}}$", "min"))
    ax.set_ylabel(axis_label("Signed action count", r"$N_{\mathrm{act}}$", "-"))
    ax.set_xticks(x)
    ax.set_xticklabels(step_summary["label"])
    style_axis(ax)
    add_panel_label(ax, "(d)")
    ax.legend(loc="lower right", frameon=False, fontsize=7.2, ncol=2)

    handles = [
        Line2D([0], [0], color=DEFAULT_PALETTE["neutral_dark"], linestyle=(0, (4, 2)), linewidth=1.15, label="Reference"),
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            color=colors[case],
            linestyle=LINESTYLES[case],
            marker=MARKERS[case],
            markersize=3.8,
            markerfacecolor="white",
            markeredgecolor=colors[case],
            linewidth=1.1,
            label=case_label(case),
        )
        for case in REPRESENTATIVE_CASES
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.55, 0.995),
        title="Representative dispatch cases",
        title_fontsize=7.4,
        columnspacing=0.9,
        handlelength=1.8,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.85, bottom=0.10)
    return save_figure(
        fig,
        FIG_DIR / "weekly_dispatch_and_occupancy_by_step",
        formats=("png",),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
    )


def redraw_pump_cost_and_sensitivity() -> list[Path]:
    case_data = pd.read_csv(PUMP_CASE_CSV)
    sensitivity = pd.read_csv(PUMP_SENSITIVITY_CSV)
    case_data = case_data[case_data["Case"].isin(STEP_CASES)].copy()
    case_data["label"] = case_data["Case"].map(case_label)
    colors = step_colors()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=cm_to_in(18.2, 8.0),
        gridspec_kw={"wspace": 0.22},
    )

    ax = axes[0]
    x = np.arange(len(case_data))
    width = 0.25
    metrics = [
        ("Gain before pump cost", "GainVsNoFlexBeforePump_CNY", DEFAULT_PALETTE["main"], -width),
        ("Pump-cost increment", "PumpCostIncreaseVsNoFlex_CNY_80MWx5", DEFAULT_PALETTE["contrast"], 0.0),
        ("Net gain after pump cost", "NetGainAfterPumpAtObservedPrice_CNY", DEFAULT_PALETTE["positive"], width),
    ]
    for label, col, color, offset in metrics:
        ax.bar(
            x + offset,
            case_data[col] / 1000.0,
            width=width,
            color=color,
            edgecolor="none",
            label=label,
            zorder=3,
        )
    ax.axhline(0, color=DEFAULT_PALETTE["neutral_mid"], linewidth=0.8, zorder=2)
    ax.set_title("Weekly value after pump-cost correction", fontweight="normal", fontsize=8.5, pad=3)
    ax.set_ylabel(axis_label("Weekly value", r"$V$", r"10$^3$ CNY"))
    ax.set_xlabel(axis_label("Decision step", r"$\Delta t_{\mathrm{d}}$", "min"))
    ax.set_xticks(x)
    ax.set_xticklabels(case_data["label"])
    style_axis(ax)
    add_panel_label(ax, "(a)")

    ax = axes[1]
    for case in STEP_CASES[:-1]:
        s = sensitivity[sensitivity["Case"].eq(case)].sort_values("PumpPriceMultiplier")
        if s.empty:
            continue
        ax.plot(
            s["PumpPriceMultiplier"],
            s["NetGainAfterPump_CNY"] / 1000.0,
            color=colors[case],
            linestyle=LINESTYLES[case],
            marker=MARKERS[case],
            markersize=4.0,
            markerfacecolor="white",
            markeredgecolor=colors[case],
            markeredgewidth=0.75,
            linewidth=1.25,
            label=case_label(case),
            zorder=3,
        )
    ax.axhline(0, color=DEFAULT_PALETTE["neutral_mid"], linestyle="--", linewidth=0.9, zorder=2)
    ax.set_title("Sensitivity to pump-electricity price", fontweight="normal", fontsize=8.5, pad=3)
    ax.set_xlabel(axis_label("Pump-electricity price multiplier", r"$\lambda_{\mathrm{pump}}$", "-"))
    ax.set_ylabel(axis_label("Net weekly value", r"$V_{\mathrm{net}}$", r"10$^3$ CNY"))
    ax.set_xlim(-0.18, 4.08)
    style_axis(ax)
    add_panel_label(ax, "(b)", y=0.815)

    fig.legend(
        handles=[
            Patch(facecolor=DEFAULT_PALETTE["main"], edgecolor="none", label="Gain before pump cost"),
            Patch(facecolor=DEFAULT_PALETTE["contrast"], edgecolor="none", label="Pump-cost increment"),
            Patch(facecolor=DEFAULT_PALETTE["positive"], edgecolor="none", label="Net gain after pump cost"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.08, 0.995),
        frameon=False,
        ncol=3,
        fontsize=7.2,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    axes[1].legend(
        loc="lower left",
        frameon=False,
        ncol=2,
        fontsize=7.0,
        handlelength=1.7,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.17)
    return save_figure(
        fig,
        FIG_DIR / "pump_cost_and_price_sensitivity",
        formats=("png",),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
    )


def redraw_fullseason_replay_and_room_deviation() -> list[Path]:
    tracking = pd.read_csv(TRACKING_CSV)
    room_series = pd.read_csv(ROOM_SERIES_CSV)
    room_summary = pd.read_csv(ROOM_SUMMARY_CSV)
    colors = {"NoFlex": DEFAULT_PALETTE["neutral_mid"], "Step15min": DEFAULT_PALETTE["main"]}

    fig, axes = plt.subplots(
        2,
        2,
        figsize=cm_to_in(18.2, 12.4),
        gridspec_kw={"wspace": 0.20, "hspace": 0.34},
    )

    ax = axes[0, 0]
    for case in ["NoFlex", "Step15min"]:
        s = tracking[tracking["Case"].eq(case)].sort_values("time_h")
        ax.plot(
            s["time_h"] / 24.0,
            s["source_T_C"],
            color=colors[case],
            linestyle="-" if case == "Step15min" else (0, (4, 2)),
            linewidth=1.05,
            label=case_label(case),
            zorder=3,
        )
    expected = tracking[tracking["Case"].eq("NoFlex")].sort_values("time_h")
    ax.plot(
        expected["time_h"] / 24.0,
        expected["expected_source_T_C"],
        color=DEFAULT_PALETTE["contrast"],
        linestyle=":",
        linewidth=1.0,
        label="Expected baseline",
        zorder=2,
    )
    ax.set_title("Supply-temperature replay tracking", fontweight="normal", fontsize=8.5, pad=3)
    ax.set_ylabel(axis_label("Supply temperature", r"$T_{\mathrm{s}}$", "℃"))
    style_axis(ax)
    add_panel_label(ax, "(a)")

    ax = axes[0, 1]
    for case in ["NoFlex", "Step15min"]:
        s = tracking[tracking["Case"].eq(case)].sort_values("time_h")
        ax.plot(
            s["time_h"] / 24.0,
            s["pump_power_W"] / 1000.0,
            color=colors[case],
            linestyle="-" if case == "Step15min" else (0, (4, 2)),
            linewidth=1.05,
            label=case_label(case),
            zorder=3,
        )
    ax.set_title("Pump-power proxy during replay", fontweight="normal", fontsize=8.5, pad=3)
    ax.set_ylabel(axis_label("Pump power", r"$P_{\mathrm{pump}}$", "kW"))
    style_axis(ax)
    add_panel_label(ax, "(b)")

    ax = axes[1, 0]
    x = room_series["time_h"] / 24.0
    ax.fill_between(
        x,
        room_series["NoFlex_room_min_C"],
        room_series["NoFlex_room_max_C"],
        color=DEFAULT_PALETTE["neutral_mid"],
        alpha=0.12,
        edgecolor="none",
        label="NoFlex range",
        zorder=1,
    )
    ax.fill_between(
        x,
        room_series["Step15min_room_min_C"],
        room_series["Step15min_room_max_C"],
        color=DEFAULT_PALETTE["main"],
        alpha=0.12,
        edgecolor="none",
        label="15 min range",
        zorder=1,
    )
    ax.plot(
        x,
        room_series["NoFlex_room_mean_C"],
        color=DEFAULT_PALETTE["neutral_mid"],
        linestyle=(0, (4, 2)),
        linewidth=1.0,
        label="NoFlex mean",
        zorder=3,
    )
    ax.plot(
        x,
        room_series["Step15min_room_mean_C"],
        color=DEFAULT_PALETTE["main"],
        linewidth=1.0,
        label="15 min mean",
        zorder=3,
    )
    ax.axhline(16, color=DEFAULT_PALETTE["neutral_dark"], linestyle=":", linewidth=0.9, zorder=2)
    ax.text(
        x.iloc[-1] * 0.985,
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
    ax.set_title("Room-temperature envelope", fontweight="normal", fontsize=8.5, pad=3)
    ax.set_xlabel(axis_label("Heating-season time", r"$t$", "d"))
    ax.set_ylabel(axis_label("Room temperature", r"$T_{\mathrm{room}}$", "℃"))
    ax.set_ylim(15.75, 24.35)
    style_axis(ax)
    add_panel_label(ax, "(c)")

    ax = axes[1, 1]
    m = room_summary[room_summary["Case"].isin(["NoFlex", "Step15min"])].copy()
    m["label"] = m["Case"].map(case_label)
    bx = np.arange(len(m))
    ax.bar(
        bx,
        m["room_min_C"],
        width=0.56,
        color=[colors[case] for case in m["Case"]],
        edgecolor="none",
        zorder=3,
    )
    ax.axhline(16, color=DEFAULT_PALETTE["neutral_dark"], linestyle=":", linewidth=0.9, zorder=2)
    for xpos, value in zip(bx, m["room_min_C"]):
        ax.text(
            xpos,
            value + 0.08,
            f"{value:.2f} \N{DEGREE CELSIUS}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=DEFAULT_PALETTE["neutral_dark"],
            bbox=PANEL_BBOX,
            zorder=4,
        )
    ax.text(
        len(bx) - 0.54,
        16.08,
        "16 \N{DEGREE CELSIUS}",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=DEFAULT_PALETTE["neutral_dark"],
        bbox=PANEL_BBOX,
        zorder=4,
    )
    ax.set_title("Severe under-supply check", fontweight="normal", fontsize=8.5, pad=3)
    ax.set_xlabel("Replay case")
    ax.set_ylabel(axis_label("Minimum room temperature", r"$T_{\mathrm{room,min}}$", "\N{DEGREE CELSIUS}"))
    ax.set_xticks(bx)
    ax.set_xticklabels(m["label"])
    ax.set_ylim(15.5, max(m["room_min_C"]) + 0.85)
    style_axis(ax)
    add_panel_label(ax, "(d)")

    handles = [
        Line2D([0], [0], color=DEFAULT_PALETTE["neutral_mid"], linestyle=(0, (4, 2)), linewidth=1.05, label="NoFlex"),
        Line2D([0], [0], color=DEFAULT_PALETTE["main"], linestyle="-", linewidth=1.05, label="15 min"),
        Line2D([0], [0], color=DEFAULT_PALETTE["contrast"], linestyle=":", linewidth=1.0, label="Expected baseline"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.55, 0.995),
        title="Full-season replay cases",
        title_fontsize=7.4,
        columnspacing=1.1,
        handlelength=2.0,
        handletextpad=0.40,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.85, bottom=0.10)
    return save_figure(
        fig,
        FIG_DIR / "fullseason_replay_and_room_deviation",
        formats=("png",),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
    )


def main() -> None:
    set_energy_style(base_size=8.0, axis_label_size=8.6, legend_size=7.2)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    copy_figure_data(
        [
            BOUNDARY_CSV,
            WEEKLY_SUMMARY_CSV,
            WEEKLY_OCCUPANCY_CSV,
            PUMP_CASE_CSV,
            PUMP_SENSITIVITY_CSV,
            TRACKING_CSV,
            ROOM_SERIES_CSV,
            ROOM_SUMMARY_CSV,
        ],
        DATA_DIR,
    )
    for schedule_file in WEEKLY_DIR.glob("schedule_*.csv"):
        copy_figure_data([schedule_file], DATA_DIR)

    outputs = {
        "curve_signed_boundary_vs_duration_selected_ambient": redraw_duration_boundary(),
        "weekly_dispatch_and_occupancy_by_step": redraw_weekly_dispatch_and_occupancy(),
        "pump_cost_and_price_sensitivity": redraw_pump_cost_and_sensitivity(),
        "fullseason_replay_and_room_deviation": redraw_fullseason_replay_and_room_deviation(),
    }

    write_manifest(
        PACKAGE_DIR / "remaining_result_figures_redraw_manifest.json",
        {
            "backend": "python/matplotlib",
            "source_data": [
                str(BOUNDARY_CSV),
                str(WEEKLY_DIR),
                str(PUMP_CASE_CSV),
                str(PUMP_SENSITIVITY_CSV),
                str(TRACKING_CSV),
                str(ROOM_SERIES_CSV),
                str(ROOM_SUMMARY_CSV),
            ],
            "exports": {
                name: [str(path.relative_to(PACKAGE_DIR)) for path in paths]
                for name, paths in outputs.items()
            },
            "palette_mode": "high-impact-order for duration, decision-step, and case order",
            "series_order": {
                "boundary_ambient_temperature": [f"{temp:+d} ℃" for temp in AMBIENT_POINTS_C],
                "decision_steps": [case_label(case) for case in STEP_CASES],
                "representative_dispatch_cases": [case_label(case) for case in REPRESENTATIVE_CASES],
            },
            "merged_or_superseded_panels": {
                "weekly_dispatch_heat_by_decision_step": "merged into weekly_dispatch_and_occupancy_by_step",
                "occupancy_state_trajectory_by_decision_step": "merged into weekly_dispatch_and_occupancy_by_step",
                "dispatch_gain_vs_pump_cost_by_step": "merged into pump_cost_and_price_sensitivity",
                "net_gain_vs_pump_price_multiplier": "merged into pump_cost_and_price_sensitivity",
                "fullseason_15min_replay_tracking_samples": "merged into fullseason_replay_and_room_deviation",
                "room_temperature_deviation_comparison": "superseded by a 16 degC severe under-supply check in fullseason_replay_and_room_deviation",
            },
            "axis_label_audit": (
                "Visible axes use complete quantity names, mathematical symbols, "
                "and publication units; no degC, MWth, MWe, or source-column tokens "
                "are used as figure-facing labels."
            ),
            "information_density_decision": (
                "Remaining low-density standalone panels are consolidated into "
                "four evidence figures: duration law, weekly dispatch and occupancy, "
                "pump sensitivity, and full-season replay with a severe under-supply check."
            ),
            "annotation_readability_audit": (
                "Panel labels are placed inside axes with light text boxes, legends "
                "are outside or in low-density corners, and dense curves avoid value labels."
            ),
            "claim_supported": (
                "The remaining figures support duration-dependent boundary behavior, "
                "decision-step dispatch behavior, pump-cost sensitivity, and model-based "
                "full-season replay executability."
            ),
            "what_this_does_not_prove": (
                "The figures do not provide independent field validation, calibrated pump "
                "electricity, or universal decision-step optimality."
            ),
        },
    )


if __name__ == "__main__":
    main()
