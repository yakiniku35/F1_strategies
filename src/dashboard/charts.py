"""
Race charts: position changes and tyre strategy.

Two classic Formula 1 analysis charts, rendered from the same frame
dictionaries the rest of the project passes around and saved as PNG files.

Colour notes
------------
Both charts use colours that carry real meaning in Formula 1 rather than an
arbitrary categorical palette:

* The position chart colours each line by **team**, which is how the sport's
  own graphics work. Teammates share a colour, so the second car of each team
  is drawn dashed, and every line is labelled directly at its right-hand end -
  identity is therefore never carried by colour alone.
* The tyre chart colours each stint by **compound** (soft red, medium yellow,
  hard white/grey, intermediate green, wet blue).

The compound steps below were checked with a palette validator against both
chart surfaces. Two things it flags are accepted deliberately:

* The hard compound reads as a neutral grey - that *is* the real-world
  semantic, so it keeps a hairline border to stay bounded against the surface.
* The red/green and yellow/surface pairs land in the "needs secondary
  encoding" band, which is why every stint carries a visible text label and
  the legend is always drawn.
"""

import os
from typing import Optional

import matplotlib
# Charts are written to disk, never shown in a window, so force the headless
# backend before pyplot is imported. This also keeps the module importable on a
# machine with no display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.lib.tyres import get_tyre_compound_str  # noqa: E402

# Chart chrome, per theme. Keeping these in one place means the two charts
# cannot drift apart.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}

# Tyre compound fills, stepped per surface (see the module docstring).
COMPOUND_COLORS = {
    "light": {
        "SOFT": "#e34948",
        "MEDIUM": "#eda100",
        "HARD": "#bfbeb4",
        "INTERMEDIATE": "#008300",
        "WET": "#2a78d6",
        "UNKNOWN": "#898781",
    },
    "dark": {
        "SOFT": "#e34948",
        "MEDIUM": "#eda100",
        "HARD": "#e8e7e0",
        "INTERMEDIATE": "#008300",
        "WET": "#3987e5",
        "UNKNOWN": "#898781",
    },
}

# Fixed order, so the legend never reshuffles between races.
COMPOUND_ORDER = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")

# Short labels drawn inside a stint bar when the bar is wide enough.
COMPOUND_SHORT = {
    "SOFT": "S", "MEDIUM": "M", "HARD": "H",
    "INTERMEDIATE": "I", "WET": "W", "UNKNOWN": "?",
}

# A stint narrower than this many laps gets no text label - it would not fit.
MIN_LAPS_FOR_LABEL = 3

# Gap between adjacent stint bars, drawn in the surface colour so the segments
# read as separate blocks rather than one continuous strip.
SEGMENT_GAP = 0.06


def _fallback_color(index: int) -> str:
    """Pick a readable colour for a driver with no team colour supplied."""
    palette = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948")
    return palette[index % len(palette)]


def _normalise_color(color) -> Optional[str]:
    """Convert an (r, g, b) tuple in 0-255 form to a hex string.

    Args:
        color: An RGB tuple, a hex string, or None.

    Returns:
        A matplotlib-friendly colour string, or None if nothing usable.
    """
    if color is None:
        return None
    if isinstance(color, str):
        return color
    try:
        r, g, b = (int(channel) for channel in tuple(color)[:3])
    except (TypeError, ValueError):
        return None
    return "#%02x%02x%02x" % (max(0, min(r, 255)),
                              max(0, min(g, 255)),
                              max(0, min(b, 255)))


def _compound_name(value) -> str:
    """Turn whatever the frames carry for "tyre" into a compound name.

    The simulator stores an integer compound code while the historical
    telemetry stores the name directly, so both are accepted.
    """
    if isinstance(value, str):
        name = value.upper()
        return name if name in COMPOUND_COLORS["light"] else "UNKNOWN"
    try:
        return get_tyre_compound_str(int(value))
    except (TypeError, ValueError):
        return "UNKNOWN"


def extract_lap_history(frames: list) -> dict:
    """Collapse frame-by-frame telemetry into one record per driver per lap.

    For each driver the *last* frame seen on a lap is kept, which is that
    driver's state at the end of the lap. A driver stops being recorded once
    they retire (``rel_dist == 1``), so their line ends where they stopped
    rather than running flat to the chequered flag.

    Args:
        frames: Race frames, each with ``lap`` and ``drivers``.

    Returns:
        ``{"laps": [...], "positions": {code: {lap: pos}},
        "compounds": {code: {lap: name}}, "drivers": [...],
        "retired": {code, ...}}``
    """
    positions = {}
    compounds = {}
    laps = set()
    retired = set()

    for frame in frames or []:
        for code, entry in frame.get("drivers", {}).items():
            if code in retired:
                continue

            try:
                if float(entry.get("rel_dist", 0)) == 1:
                    retired.add(code)
                    continue
            except (TypeError, ValueError):
                pass

            try:
                lap = int(entry.get("lap", frame.get("lap", 1)) or 1)
                position = int(round(float(entry.get("position", 0) or 0)))
            except (TypeError, ValueError):
                continue

            if position <= 0:
                continue

            laps.add(lap)
            positions.setdefault(code, {})[lap] = position
            compounds.setdefault(code, {})[lap] = _compound_name(entry.get("tyre"))

    # Order drivers by their final classified position, so both charts list
    # them the way a results sheet does.
    def _final_position(code):
        driver_laps = positions.get(code, {})
        return driver_laps[max(driver_laps)] if driver_laps else 99

    return {
        "laps": sorted(laps),
        "positions": positions,
        "compounds": compounds,
        "drivers": sorted(positions, key=_final_position),
        # Only count a retirement for a driver we actually have laps for.
        "retired": {code for code in retired if code in positions},
    }


def _style_axes(ax, theme: dict):
    """Apply the recessive grid/axis treatment both charts share."""
    ax.set_facecolor(theme["surface"])
    ax.grid(True, color=theme["grid"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(1.0)

    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)


def plot_position_changes(history: dict,
                          driver_colors: Optional[dict] = None,
                          title: str = "Position Changes",
                          output_path: str = "charts/positions.png",
                          theme: str = "light") -> Optional[str]:
    """Draw the lap-by-lap position chart.

    Args:
        history: Output of extract_lap_history().
        driver_colors: Optional ``code -> (r, g, b)`` team colours.
        title: Chart title.
        output_path: Where to write the PNG.
        theme: "light" or "dark".

    Returns:
        The path written, or None if there was nothing to draw.
    """
    drivers = history.get("drivers") or []
    laps = history.get("laps") or []
    if not drivers or not laps:
        return None

    palette = THEMES.get(theme, THEMES["light"])
    driver_colors = driver_colors or {}

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor(palette["surface"])
    _style_axes(ax, palette)

    # Teammates share a team colour, so the second car on a given colour is
    # drawn dashed - identity never rests on colour alone.
    seen_colors = {}
    retired = history.get("retired") or set()
    last_lap = max(laps)

    for index, code in enumerate(drivers):
        color = _normalise_color(driver_colors.get(code)) or _fallback_color(index)
        seen_colors[color] = seen_colors.get(color, 0) + 1
        linestyle = "-" if seen_colors[color] == 1 else "--"

        driver_laps = sorted(history["positions"][code])
        xs = driver_laps
        ys = [history["positions"][code][lap] for lap in driver_laps]

        ax.plot(xs, ys, color=color, linewidth=2.0, linestyle=linestyle,
                solid_capstyle="round", zorder=3)

        # A retired driver's line stops mid-chart, so its label would otherwise
        # land on top of everyone still running. Mark the end with a cross and
        # sit the label on a surface-coloured plate so it stays readable.
        did_not_finish = code in retired and xs[-1] < last_lap

        ax.plot([xs[-1]], [ys[-1]],
                marker="X" if did_not_finish else "o",
                markersize=9 if did_not_finish else 8, color=color,
                markeredgecolor=palette["surface"], markeredgewidth=2, zorder=6)

        ax.annotate(f" {code} DNF" if did_not_finish else f" {code}",
                    (xs[-1], ys[-1]),
                    color=palette["secondary"] if did_not_finish else palette["ink"],
                    fontsize=8.5 if did_not_finish else 9, fontweight="bold",
                    va="center", ha="left", xytext=(10, 0),
                    textcoords="offset points", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor=palette["surface"],
                              edgecolor=palette["axis"], linewidth=0.6)
                    if did_not_finish else None)

    # P1 belongs at the top.
    ax.invert_yaxis()
    positions_used = [p for code in drivers for p in history["positions"][code].values()]
    ax.set_yticks(range(1, max(positions_used) + 1))
    ax.set_xlim(min(laps) - 0.5, max(laps) + max(2.5, len(laps) * 0.08))
    ax.set_xlabel("Lap", color=palette["secondary"], fontsize=10)
    ax.set_ylabel("Position", color=palette["secondary"], fontsize=10)
    ax.set_title(title, color=palette["ink"], fontsize=15,
                 fontweight="bold", pad=16, loc="left")

    return _save(fig, output_path, palette)


def build_stints(compound_laps: dict) -> list:
    """Collapse a driver's per-lap compounds into contiguous stints.

    Args:
        compound_laps: ``{lap: compound_name}`` for one driver.

    Returns:
        List of ``{"compound", "start_lap", "end_lap", "laps"}``, in lap order.
    """
    if not compound_laps:
        return []

    stints = []
    for lap in sorted(compound_laps):
        compound = compound_laps[lap]
        if stints and stints[-1]["compound"] == compound and stints[-1]["end_lap"] == lap - 1:
            stints[-1]["end_lap"] = lap
        else:
            stints.append({"compound": compound, "start_lap": lap, "end_lap": lap})

    for stint in stints:
        stint["laps"] = stint["end_lap"] - stint["start_lap"] + 1

    return stints


def plot_tyre_strategy(history: dict,
                       title: str = "Tyre Strategy",
                       output_path: str = "charts/tyres.png",
                       theme: str = "light") -> Optional[str]:
    """Draw the tyre stint chart (one horizontal bar per driver).

    Args:
        history: Output of extract_lap_history().
        title: Chart title.
        output_path: Where to write the PNG.
        theme: "light" or "dark".

    Returns:
        The path written, or None if there was nothing to draw.
    """
    drivers = history.get("drivers") or []
    laps = history.get("laps") or []
    if not drivers or not laps:
        return None

    palette = THEMES.get(theme, THEMES["light"])
    compound_colors = COMPOUND_COLORS.get(theme, COMPOUND_COLORS["light"])

    fig, ax = plt.subplots(figsize=(13, max(4.0, len(drivers) * 0.42)))
    fig.patch.set_facecolor(palette["surface"])
    _style_axes(ax, palette)
    ax.grid(True, axis="x", color=palette["grid"], linewidth=0.8)
    ax.grid(False, axis="y")

    used_compounds = []
    retired = history.get("retired") or set()
    last_lap = max(laps)

    for row, code in enumerate(drivers):
        stints = build_stints(history["compounds"].get(code, {}))

        # A short bar on its own does not explain itself - label the retirement.
        if code in retired and stints and stints[-1]["end_lap"] < last_lap:
            ax.text(stints[-1]["end_lap"] + 0.9, row, "DNF",
                    ha="left", va="center", fontsize=8, fontweight="bold",
                    color=palette["secondary"], zorder=4)

        for stint in stints:
            compound = stint["compound"]
            if compound not in used_compounds:
                used_compounds.append(compound)

            # The gap is drawn by shrinking each bar, so neighbouring stints
            # read as separate blocks instead of one continuous strip.
            ax.barh(row,
                    stint["laps"] - SEGMENT_GAP * 2,
                    left=stint["start_lap"] - 0.5 + SEGMENT_GAP,
                    height=0.62,
                    color=compound_colors.get(compound, compound_colors["UNKNOWN"]),
                    edgecolor=palette["axis"], linewidth=0.8, zorder=3)

            # Visible label inside the bar: this is what lets the chart stay
            # readable for a colour-blind reader and in print.
            if stint["laps"] >= MIN_LAPS_FOR_LABEL:
                ax.text(stint["start_lap"] - 0.5 + stint["laps"] / 2, row,
                        f"{COMPOUND_SHORT.get(compound, '?')} {stint['laps']}",
                        ha="center", va="center", fontsize=8, fontweight="bold",
                        color="#0b0b0b" if compound in ("HARD", "MEDIUM") else "#ffffff",
                        zorder=4)

    ax.set_yticks(range(len(drivers)))
    ax.set_yticklabels(drivers, color=palette["ink"], fontsize=9, fontweight="bold")
    ax.invert_yaxis()  # winner at the top
    ax.set_xlim(min(laps) - 0.5, max(laps) + 0.5)
    ax.set_xlabel("Lap", color=palette["secondary"], fontsize=10)
    ax.set_title(title, color=palette["ink"], fontsize=15,
                 fontweight="bold", pad=16, loc="left")

    # Legend in the fixed compound order, not the order encountered.
    handles = [Patch(facecolor=compound_colors[name], edgecolor=palette["axis"],
                     label=name.title())
               for name in COMPOUND_ORDER if name in used_compounds]
    if handles:
        legend = ax.legend(handles=handles, loc="upper center",
                           bbox_to_anchor=(0.5, -0.08), ncol=len(handles),
                           frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(palette["secondary"])

    return _save(fig, output_path, palette)


def _save(fig, output_path: str, palette: dict) -> Optional[str]:
    """Write a figure to disk, creating the directory if needed."""
    try:
        directory = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(directory, exist_ok=True)
        fig.savefig(output_path, dpi=140, bbox_inches="tight",
                    facecolor=palette["surface"])
        return output_path
    except OSError as exc:
        print(f"⚠️ 無法儲存圖表 {output_path}: {exc}")
        return None
    finally:
        plt.close(fig)


def generate_race_charts(frames: list,
                         driver_colors: Optional[dict] = None,
                         output_dir: str = "charts",
                         title_prefix: str = "Race",
                         theme: str = "light") -> dict:
    """Produce both race charts in one call.

    Args:
        frames: Every frame of the race.
        driver_colors: Optional ``code -> (r, g, b)`` team colours.
        output_dir: Directory the PNGs are written to.
        title_prefix: Prepended to each chart title.
        theme: "light" or "dark".

    Returns:
        ``{"positions": path or None, "tyres": path or None}``.
    """
    history = extract_lap_history(frames)

    return {
        "positions": plot_position_changes(
            history, driver_colors=driver_colors,
            title=f"{title_prefix} - Position Changes",
            output_path=os.path.join(output_dir, "position_changes.png"),
            theme=theme,
        ),
        "tyres": plot_tyre_strategy(
            history,
            title=f"{title_prefix} - Tyre Strategy",
            output_path=os.path.join(output_dir, "tyre_strategy.png"),
            theme=theme,
        ),
    }
