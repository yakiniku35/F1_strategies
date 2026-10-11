"""Tests for the static strategy page's data file (scripts/build_web_data.py)."""

import json
import math

import pytest

from scripts import build_web_data
from src.strategy_analyzer import StrategyAnalyzer


@pytest.fixture(scope="module")
def data():
    return build_web_data.build()


def test_lap_times_sum_to_the_estimated_race_time():
    analyzer = StrategyAnalyzer("Silverstone", total_laps=52)
    for option in analyzer.generate_strategy_options(current_position=3):
        lap_times = analyzer.strategy_lap_times(option.pit_laps, option.compounds)
        assert len(lap_times) == 52
        assert math.isclose(sum(lap_times), option.estimated_time)


def test_pit_loss_lands_on_the_pit_lap():
    analyzer = StrategyAnalyzer("default", total_laps=10)
    lap_times = analyzer.strategy_lap_times([4], ["HARD", "HARD"])

    # Lap 4 starts the fresh stint, so it is a new-tyre lap plus the stop.
    assert lap_times[3] == pytest.approx(lap_times[0] + analyzer.PIT_STOP_TIME_LOSS)
    assert lap_times[2] > lap_times[0]   # worn tyres before the stop


def test_every_bundled_race_is_exported(data):
    year, schedule = build_web_data.load_fallback_schedule()
    assert data["season"] == year
    assert [r["round"] for r in data["races"]] == [r["round"] for r in schedule]


def test_strategies_are_sorted_and_carry_a_full_trace(data):
    for race in data["races"]:
        totals = [s["total_time"] for s in race["strategies"]]
        assert totals == sorted(totals)
        for strategy in race["strategies"]:
            assert len(strategy["lap_times"]) == race["laps"]
            assert len(strategy["compounds"]) == len(strategy["pit_laps"]) + 1
            assert sum(strategy["lap_times"]) == pytest.approx(strategy["total_time"], abs=0.05)


def test_midfield_view_never_loses_every_strategy(data):
    for race in data["races"]:
        assert any(not s["front_runner_only"] for s in race["strategies"])
        assert sum(s["front_runner_only"] for s in race["strategies"]) == 1


def test_circuits_resolve_to_the_analyzer_names(data):
    by_location = {r["location"]: r for r in data["races"]}
    assert by_location["Silverstone"]["tyre_stress"] == "high"
    assert by_location["Monza"]["tyre_stress"] == "low"
    # Monaco is filed under its GP name, not "Monte Carlo".
    assert by_location["Monte Carlo"]["analyzer_track"] == "Monaco"
    assert by_location["Monte Carlo"]["base_lap_time"] == 75.0


def test_undercut_grid_matches_the_analyzer(data):
    grid = data["undercut"]
    analyzer = StrategyAnalyzer("default", total_laps=100)
    for gi, gap in enumerate(grid["gaps"]):
        for ai, diff in enumerate(grid["age_diffs"]):
            expected = analyzer.analyze_undercut_opportunity(45, gap, 0, diff)
            score, viable, level = grid["cells"]["in_window"][gi][ai]
            assert score == expected["score"]
            assert viable == int(expected["viable"])
            assert level in {"strong", "possible", "risky", "none"}


def test_main_writes_compact_json(tmp_path):
    out = tmp_path / "strategy.json"
    build_web_data.main(["-o", str(out)])

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["races"]
    assert out.stat().st_size < 200 * 1024
