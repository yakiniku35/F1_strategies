#!/usr/bin/env python
"""
Build the data file for the static strategy page in ``web/``.

The page itself does no strategy maths: every number it shows is computed here
by ``src/strategy_analyzer.py`` and written to ``web/data/strategy.json``, so
the Python code stays the single source of truth.

Usage:
    python scripts/build_web_data.py            # writes web/data/strategy.json
    python scripts/build_web_data.py -o out.json
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.strategy_analyzer import StrategyAnalyzer  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "web" / "data" / "strategy.json"

# Undercut lookup grid. The page reads its sliders off this table instead of
# re-implementing analyze_undercut_opportunity() in JavaScript.
UNDERCUT_GAPS = [round(0.5 * i, 1) for i in range(13)]   # 0.0 - 6.0 s
UNDERCUT_AGE_DIFFS = list(range(0, 11))                  # 0 - 10 laps fresher

# The leading marker of each undercut recommendation, mapped to a level the
# page can colour and translate.
UNDERCUT_LEVELS = {"🟢": "strong", "🟡": "possible", "🟠": "risky", "🔴": "none"}


def load_fallback_schedule():
    """Return the bundled season calendar without importing FastF1.

    ``src/simulation/__init__.py`` pulls in modules that import FastF1 at module
    scope, so the provider module is loaded straight from its file instead of
    through the package. It only needs pandas.
    """
    path = REPO_ROOT / "src" / "simulation" / "future_race_data.py"
    spec = importlib.util.spec_from_file_location("_future_race_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provider = module.FutureRaceDataProvider
    return provider.FALLBACK_YEAR, [race.copy() for race in provider.FALLBACK_SCHEDULE]


def analyzer_track_key(race: dict) -> str:
    """Pick the name StrategyAnalyzer knows this circuit by.

    The analyzer's tables mix city names (Silverstone, Monza) with country
    names (Bahrain, Austria), so try the location first, then the GP name.
    """
    known = set(StrategyAnalyzer.TRACK_BASE_LAP_TIMES)
    probe = StrategyAnalyzer("default", race["laps"])
    for candidate in (race["location"], race["gp"]):
        if candidate in known:
            return candidate
        probe.track_name = candidate
        if probe._get_track_tyre_stress() != StrategyAnalyzer.TRACK_TYRE_STRESS["medium"]:
            return candidate
    return race["location"]


def stress_label(stress: float) -> str:
    for label, value in StrategyAnalyzer.TRACK_TYRE_STRESS.items():
        if value == stress:
            return label
    return "medium"


def build_race(race: dict) -> dict:
    """Strategy options and their lap-by-lap trace for one Grand Prix."""
    key = analyzer_track_key(race)
    analyzer = StrategyAnalyzer(track_name=key, total_laps=race["laps"])

    # A front-runner (position <= 5) also gets the "Undercut Special" option;
    # the midfield list is the same set without it.
    front = analyzer.generate_strategy_options(current_position=3)
    midfield_names = {s.name for s in analyzer.generate_strategy_options(current_position=10)}

    strategies = []
    for option in front:
        lap_times = analyzer.strategy_lap_times(option.pit_laps, option.compounds)
        strategies.append({
            "name": option.name,
            "stops": option.stops,
            "pit_laps": option.pit_laps,
            "compounds": option.compounds,
            "risk": option.risk_level,
            "total_time": round(option.estimated_time, 3),
            "front_runner_only": option.name not in midfield_names,
            "lap_times": [round(t, 3) for t in lap_times],
        })

    fuel = analyzer.simulate_fuel_strategy()

    return {
        "round": race["round"],
        "name": race["name"],
        "gp": race["gp"],
        "location": race["location"],
        "date": race["date"],
        "laps": race["laps"],
        "analyzer_track": key,
        "tyre_stress": stress_label(analyzer.tyre_stress),
        "base_lap_time": StrategyAnalyzer.TRACK_BASE_LAP_TIMES.get(
            key, StrategyAnalyzer.TRACK_BASE_LAP_TIMES["default"]),
        "strategies": strategies,
        "fuel": {
            "per_lap_kg": fuel["fuel_per_lap"],
            "initial_penalty_s": fuel["initial_penalty"],
            "penalty_by_lap": [lap["fuel_penalty"] for lap in fuel["lap_times"]],
        },
    }


def build_undercut_grid() -> dict:
    """Undercut score for every gap / tyre-age combination, in and out of the pit window."""
    analyzer = StrategyAnalyzer("default", total_laps=100)
    windows = {"in_window": 45, "out_of_window": 10}   # lap 45/100 is inside 30-60 %

    grid = {}
    for label, lap in windows.items():
        rows = []
        for gap in UNDERCUT_GAPS:
            row = []
            for diff in UNDERCUT_AGE_DIFFS:
                result = analyzer.analyze_undercut_opportunity(
                    current_lap=lap, gap_to_car_ahead=gap,
                    our_tyre_age=0, their_tyre_age=diff,
                )
                level = UNDERCUT_LEVELS.get(result["recommendation"][:1], "none")
                row.append([result["score"], int(result["viable"]), level])
            rows.append(row)
        grid[label] = rows

    return {
        "gaps": UNDERCUT_GAPS,
        "age_diffs": UNDERCUT_AGE_DIFFS,
        "max_gap": StrategyAnalyzer.MAX_UNDERCUT_GAP,
        "cells": grid,
    }


def build() -> dict:
    year, schedule = load_fallback_schedule()
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season": year,
        "model": {
            "pit_loss_s": StrategyAnalyzer.PIT_STOP_TIME_LOSS,
            "compound_pace_s": StrategyAnalyzer.COMPOUND_PACE,
            "compound_degradation_s": StrategyAnalyzer.COMPOUND_DEGRADATION,
            "tyre_stress": StrategyAnalyzer.TRACK_TYRE_STRESS,
        },
        "races": [build_race(race) for race in schedule],
        "undercut": build_undercut_grid(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output file (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})")
    args = parser.parse_args(argv)

    data = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                           encoding="utf-8")
    size_kb = args.output.stat().st_size / 1024
    print(f"Wrote {args.output} ({len(data['races'])} races, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
