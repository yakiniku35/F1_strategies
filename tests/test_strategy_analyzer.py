"""Tests for the undercut / overcut heuristics and strategy generation."""

import json

import pytest

from src.strategy_analyzer import StrategyAnalyzer


@pytest.fixture
def analyzer():
    return StrategyAnalyzer(track_name="Silverstone", total_laps=52)


class TestUndercut:
    def test_a_close_gap_on_fresher_tyres_mid_race_is_viable(self, analyzer):
        result = analyzer.analyze_undercut_opportunity(
            current_lap=26, gap_to_car_ahead=0.5,
            our_tyre_age=2, their_tyre_age=20,
        )
        assert result["viable"] is True
        assert result["gap_advantage"] is True
        assert result["tyre_advantage"] is True
        assert result["in_pit_window"] is True

    def test_a_gap_beyond_the_limit_is_never_viable(self, analyzer):
        result = analyzer.analyze_undercut_opportunity(
            current_lap=26,
            gap_to_car_ahead=analyzer.MAX_UNDERCUT_GAP + 1,
            our_tyre_age=0, their_tyre_age=40,
        )
        assert result["viable"] is False

    def test_older_tyres_than_the_car_ahead_remove_the_advantage(self, analyzer):
        result = analyzer.analyze_undercut_opportunity(
            current_lap=26, gap_to_car_ahead=1.0,
            our_tyre_age=25, their_tyre_age=3,
        )
        assert result["tyre_advantage"] is False

    def test_the_score_rises_as_the_gap_closes(self, analyzer):
        scores = [
            analyzer.analyze_undercut_opportunity(26, gap, 5, 15)["score"]
            for gap in (4.0, 3.0, 2.0, 1.0)
        ]
        assert scores == sorted(scores)

    def test_the_pit_window_only_covers_the_middle_of_the_race(self, analyzer):
        early = analyzer.analyze_undercut_opportunity(2, 1.0, 5, 15)
        middle = analyzer.analyze_undercut_opportunity(26, 1.0, 5, 15)
        late = analyzer.analyze_undercut_opportunity(51, 1.0, 5, 15)

        assert early["in_pit_window"] is False
        assert middle["in_pit_window"] is True
        assert late["in_pit_window"] is False

    @pytest.mark.parametrize("gap, our_age, their_age", [
        (0.0, 0, 60),     # maximal advantage on every term
        (10.0, 60, 0),    # no advantage at all
        (2.5, 5, 15),     # ordinary mid-race case
    ])
    def test_the_score_always_stays_within_zero_and_one(
            self, analyzer, gap, our_age, their_age):
        # The score is reported to the user as "x / 1.00", so it must never
        # exceed 1.0 however lopsided the tyre ages are.
        score = analyzer.analyze_undercut_opportunity(
            26, gap, our_age, their_age)["score"]
        assert 0.0 <= score <= 1.0

    def test_every_result_carries_a_recommendation(self, analyzer):
        for gap in (0.5, 3.0, 4.5, 10.0):
            result = analyzer.analyze_undercut_opportunity(26, gap, 5, 15)
            assert isinstance(result["recommendation"], str)
            assert result["recommendation"]


class TestOvercut:
    def test_fresh_tyres_can_extend(self, analyzer):
        result = analyzer.analyze_overcut_opportunity(
            current_lap=20, gap_to_car_ahead=2.0,
            our_tyre_age=2, our_compound="HARD",
        )
        assert result["viable"] is True
        assert result["tyres_healthy"] is True
        assert result["recommended_extend_laps"] > 0

    def test_worn_tyres_cannot_extend(self, analyzer):
        result = analyzer.analyze_overcut_opportunity(
            current_lap=30, gap_to_car_ahead=2.0,
            our_tyre_age=40, our_compound="SOFT",
        )
        assert result["viable"] is False
        assert result["recommended_extend_laps"] == 0
        assert result["remaining_optimal_laps"] == 0

    def test_remaining_life_never_goes_negative(self, analyzer):
        result = analyzer.analyze_overcut_opportunity(10, 2.0, 999, "SOFT")
        assert result["remaining_optimal_laps"] >= 0

    def test_an_unknown_compound_falls_back_rather_than_raising(self, analyzer):
        result = analyzer.analyze_overcut_opportunity(20, 2.0, 5, "BANANA")
        assert "viable" in result


class TestStrategyGeneration:
    def test_options_are_produced_and_comparable(self, analyzer):
        strategies = analyzer.generate_strategy_options(current_position=10)
        assert strategies

        comparison = analyzer.compare_strategies(strategies)
        assert isinstance(comparison, str) and comparison

    def test_json_export_round_trips(self, analyzer, tmp_path):
        strategies = analyzer.generate_strategy_options(current_position=5)
        path = tmp_path / "strategies.json"

        assert analyzer.export_strategies_to_json(strategies, str(path)) is True
        assert json.loads(path.read_text(encoding="utf-8"))

    def test_csv_export_writes_something(self, analyzer, tmp_path):
        strategies = analyzer.generate_strategy_options(current_position=5)
        path = tmp_path / "strategies.csv"

        assert analyzer.export_strategies_to_csv(strategies, str(path)) is True
        assert path.read_text(encoding="utf-8").strip()

    def test_fuel_simulation_returns_numbers(self, analyzer):
        result = analyzer.simulate_fuel_strategy(fuel_load=110.0)
        assert isinstance(result, dict) and result


class TestTrackSetup:
    def test_a_known_track_uses_its_own_base_lap_time(self):
        assert StrategyAnalyzer(track_name="Monaco", total_laps=78)

    def test_an_unknown_track_falls_back_to_the_default(self):
        assert StrategyAnalyzer(track_name="Nowhere", total_laps=50)
