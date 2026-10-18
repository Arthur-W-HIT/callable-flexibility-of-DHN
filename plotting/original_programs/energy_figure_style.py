"""Reusable Matplotlib helpers for energy manuscript result figures.

This module is intentionally small. Copy it into a project run package or import it
from the skill directory when building Python figure scripts.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
except ModuleNotFoundError:
    matplotlib = None
    plt = None
    LinearSegmentedColormap = None
    ListedColormap = None


PALETTE_NAME = "energy-results-figure high-impact default palette"
ORDERED_PALETTE_NAME = "energy-results-figure high-impact order"

DEFAULT_PALETTE = {
    "main": "#0F4D92",
    "positive": "#8BCF8B",
    "contrast": "#B64342",
    "secondary": "#42949E",
    "constraint": "#9A4D8E",
    "warning": "#D89C2B",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
    "neutral_dark": "#272727",
    "grid": "#E6E8EB",
}

DEFAULT_COLOR_ORDER = (
    DEFAULT_PALETTE["main"],
    DEFAULT_PALETTE["positive"],
    DEFAULT_PALETTE["contrast"],
    DEFAULT_PALETTE["secondary"],
    DEFAULT_PALETTE["constraint"],
    DEFAULT_PALETTE["warning"],
    DEFAULT_PALETTE["neutral_mid"],
)

FAMILY_PALETTES = {
    "high-impact-order": DEFAULT_COLOR_ORDER,
    "muted-order": (
        "#4E79A7",
        "#A0CBE8",
        "#59A14F",
        "#8CD17D",
        "#F28E2B",
        "#FFBE7D",
        "#B07AA1",
        "#D4A6C8",
        "#79706E",
        "#BAB0AC",
    ),
    "ordered-blue": (
        "#0F4D92",
        "#2E6FA8",
        "#4C8ABA",
        "#6AA5CC",
        "#88C0DE",
        "#A6D9EF",
        "#767676",
    ),
    "ordered-teal": (
        "#0B5D66",
        "#217782",
        "#42949E",
        "#63AEB7",
        "#87C7CE",
        "#B3DEE3",
        "#767676",
    ),
}

SEMANTIC_ACCENT_ROLES = {
    "baseline": "neutral_dark",
    "reference": "neutral_dark",
    "constraint": "constraint",
    "limit": "constraint",
    "warning": "warning",
    "penalty": "contrast",
    "negative": "contrast",
    "positive": "positive",
    "support": "positive",
    "zero": "neutral_mid",
}

DEFAULT_FONT_FAMILY = ("Times New Roman", "DejaVu Serif")
DEFAULT_BASE_SIZE = 8.2
DEFAULT_AXIS_LABEL_SIZE = 9.2
DEFAULT_LEGEND_SIZE = 8.0
DEFAULT_EXPORT_FORMATS = ("png", "pdf", "svg")
DEFAULT_EXPORT_DPI = 600
LINE_ART_DPI = 1000
COMBINATION_ART_DPI = 500
IMAGE_ART_DPI = 300

STATUS_COLOR_ROLES = {
    "supported": "positive",
    "pass": "positive",
    "feasible": "positive",
    "marginal": "warning",
    "marginal_screening": "warning",
    "review": "contrast",
    "sensitive": "contrast",
    "extrapolated": "contrast",
    "extrapolated_or_LP1_sensitive": "contrast",
    "constraint": "constraint",
    "inactive": "neutral_light",
}


def cm_to_in(width_cm: float, height_cm: float) -> tuple[float, float]:
    """Convert figure dimensions from centimeters to inches."""

    return width_cm / 2.54, height_cm / 2.54


def set_energy_style(
    *,
    font_family: Sequence[str] | None = None,
    base_size: float = 8.2,
    axis_label_size: float = 9.2,
    legend_size: float = 8.0,
    use_grid: bool = True,
) -> None:
    """Apply conservative defaults for energy-journal result figures."""

    _require_matplotlib()
    if font_family is None:
        font_family = DEFAULT_FONT_FAMILY

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": list(font_family),
            "font.size": base_size,
            "axes.labelsize": axis_label_size,
            "axes.titlesize": axis_label_size,
            "axes.linewidth": 0.7,
            "axes.grid": use_grid,
            "grid.color": DEFAULT_PALETTE["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.9,
            "legend.fontsize": legend_size,
            "legend.frameon": False,
            "xtick.labelsize": base_size,
            "ytick.labelsize": base_size,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "lines.linewidth": 1.35,
            "patch.linewidth": 0.7,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def add_panel_label(
    ax,
    label: str,
    *,
    x: float = 0.0,
    y: float = 1.02,
    fontsize: float = 9.0,
    fontweight: str = "normal",
) -> None:
    """Place a stable panel label using axes coordinates."""

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight=fontweight,
        color=DEFAULT_PALETTE["neutral_dark"],
    )


def math_quantity_label(symbol: str, subscript: str | None, unit: str | None) -> str:
    """Build a manuscript-style label with upright descriptor subscript and unit."""

    if subscript:
        quantity = rf"${symbol}_{{\mathrm{{{subscript}}}}}$"
    else:
        quantity = rf"${symbol}$"
    if unit:
        return rf"{quantity} ({unit})"
    return quantity


def axis_label(quantity_name: str, symbol: str | None, unit: str | None) -> str:
    """Build a full manuscript axis label: quantity name, symbol, and unit."""

    parts = [str(quantity_name).strip()]
    if symbol:
        parts.append(str(symbol).strip())
    label = ", ".join(part for part in parts if part)
    if unit:
        label = f"{label} ({str(unit).strip()})"
    return label


def math_power_unit(kind: str) -> str:
    """Return common upright MW unit strings for Matplotlib mathtext."""

    normalized = kind.strip().lower()
    if normalized in {"thermal", "th", "mwth"}:
        return r"MW$_{\mathrm{th}}$"
    if normalized in {"electric", "e", "mwe"}:
        return r"MW$_{\mathrm{e}}$"
    return kind


def set_clean_3d_axes(
    ax,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    box_aspect: tuple[float, float, float] | None = None,
    grid_color: str | None = None,
    pane_alpha: float = 0.0,
    grid_alpha: float = 0.55,
) -> None:
    """Clean Matplotlib 3D panes, grid, limits, and box aspect."""

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if zlim is not None:
        ax.set_zlim(*zlim)

    if box_aspect is None and all(v is not None for v in (xlim, ylim, zlim)):
        xr = max(xlim[1] - xlim[0], 1e-9)
        yr = max(ylim[1] - ylim[0], 1e-9)
        zr = max(zlim[1] - zlim[0], 1e-9)
        scale = max(xr, yr, zr)
        box_aspect = (xr / scale, yr / scale, zr / scale)
    if box_aspect is not None:
        ax.set_box_aspect(box_aspect)

    if grid_color is None:
        grid_color = DEFAULT_PALETTE["grid"]

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, pane_alpha))
        axis.pane.set_edgecolor(grid_color)
        try:
            axis._axinfo["grid"]["color"] = grid_color
            axis._axinfo["grid"]["linewidth"] = 0.55
            axis._axinfo["grid"]["alpha"] = grid_alpha
        except KeyError:
            pass
    ax.grid(True)


def save_figure(
    fig,
    output_base: str | Path,
    *,
    formats: Sequence[str] = DEFAULT_EXPORT_FORMATS,
    dpi: int = DEFAULT_EXPORT_DPI,
    bbox_inches: str = "tight",
    pad_inches: float = 0.03,
    close: bool = True,
) -> list[Path]:
    """Save a figure in multiple formats and optionally close it."""

    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for fmt in formats:
        fmt_clean = fmt.lower().lstrip(".")
        path = output_base.with_suffix(f".{fmt_clean}")
        save_kwargs = {"bbox_inches": bbox_inches, "pad_inches": pad_inches}
        if fmt_clean in {"png", "tif", "tiff", "jpg", "jpeg"}:
            save_kwargs["dpi"] = dpi
        fig.savefig(path, **save_kwargs)
        outputs.append(path)
    if close:
        plt.close(fig)
    return outputs


def palette_manifest() -> dict[str, object]:
    """Return the named default palette for run manifests."""

    return {
        "name": PALETTE_NAME,
        "roles": DEFAULT_PALETTE,
        "ordered_name": ORDERED_PALETTE_NAME,
        "preferred_order": list(DEFAULT_COLOR_ORDER),
        "family_palettes": {name: list(colors) for name, colors in FAMILY_PALETTES.items()},
        "semantic_accent_roles": SEMANTIC_ACCENT_ROLES,
    }


def color_order(n: int | None = None, *, mode: str = "high-impact-order") -> list[str]:
    """Return the first n colors from a named ordered palette.

    Use this for most multi-series figures. Semantic colors should be added as
    explicit accents instead of reordering the main series colors.
    """

    palette = list(FAMILY_PALETTES.get(mode, DEFAULT_COLOR_ORDER))
    if n is None:
        return palette
    if n <= len(palette):
        return palette[:n]
    repeated = [palette[i % len(palette)] for i in range(n)]
    return repeated


def assign_series_colors(
    labels: Sequence[str],
    *,
    mode: str = "high-impact-order",
    semantic_overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Assign colors by series order, with optional semantic accent overrides.

    `semantic_overrides` maps a label to either a palette role (for example
    "baseline" or "constraint") or a literal hex color. Keep overrides sparse;
    they are intended for accents, not for every plotted series.
    """

    labels_clean = [str(label) for label in labels]
    colors = color_order(len(labels_clean), mode=mode)
    assigned = dict(zip(labels_clean, colors))
    if not semantic_overrides:
        return assigned

    for label, role_or_hex in semantic_overrides.items():
        key = str(label)
        if key not in assigned:
            continue
        value = str(role_or_hex)
        if value.startswith("#"):
            assigned[key] = value
            continue
        role = SEMANTIC_ACCENT_ROLES.get(value, value)
        assigned[key] = DEFAULT_PALETTE.get(role, assigned[key])
    return assigned


def status_color(status: str, *, fallback: str = "neutral_mid") -> str:
    """Return a role color for common audit/status labels."""

    role = STATUS_COLOR_ROLES.get(str(status).strip(), fallback)
    return DEFAULT_PALETTE.get(role, DEFAULT_PALETTE[fallback])


def listed_status_cmap(statuses: Sequence[str]) -> ListedColormap:
    """Build a ListedColormap from status labels using the default palette."""

    _require_matplotlib()
    return ListedColormap([status_color(status) for status in statuses])


def sequential_cmap(name: str = "energy_seq") -> LinearSegmentedColormap:
    """Default ordered map for scalar fields; avoid rainbow/jet."""

    _require_matplotlib()
    return LinearSegmentedColormap.from_list(
        name,
        [DEFAULT_PALETTE["contrast"], "#F7F7F7", DEFAULT_PALETTE["secondary"]],
    )


def diverging_cmap(name: str = "energy_div") -> LinearSegmentedColormap:
    """Default diverging map for values with a meaningful center/reference."""

    _require_matplotlib()
    return LinearSegmentedColormap.from_list(
        name,
        [DEFAULT_PALETTE["contrast"], "#F7F7F7", DEFAULT_PALETTE["main"]],
    )


def copy_figure_data(sources: Iterable[str | Path], data_dir: str | Path) -> list[Path]:
    """Copy plotting data into a package-local figure_data directory."""

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source in sources:
        src = Path(source)
        dst = data_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def runtime_manifest() -> dict[str, object]:
    """Return Python and Matplotlib runtime metadata."""

    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "matplotlib": matplotlib.__version__ if matplotlib is not None else "not-installed",
    }


def write_manifest(path: str | Path, payload: Mapping[str, object]) -> Path:
    """Write a JSON manifest with creation time and runtime metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime": runtime_manifest(),
        "palette": palette_manifest(),
        **dict(payload),
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def luminance_text_color(hex_color: str, *, dark: str = "#333333", light: str = "white") -> str:
    """Choose black/white-like text color from a hex background."""

    value = hex_color.lstrip("#")
    red, green, blue = (int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return light if luminance < 0.5 else dark


def _require_matplotlib() -> None:
    if plt is None:
        raise RuntimeError(
            "Matplotlib is required for plotting helpers. Palette utilities can "
            "be used without Matplotlib, but plotting/export functions need it."
        )
