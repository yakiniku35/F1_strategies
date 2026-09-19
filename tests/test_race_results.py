"""Tests for the final classification and championship points."""

import json

import pytest

from src.race_results import (
    POINTS_BY_POSITION,
    build_classification,
    classification_from_frames,
    compute_lap_times,
    export_classification,
    find_fastest_lap,
    format_classification,
)

LAP_DISTANCE = 5000.0


def driver(lap, dist, speed=200.0, rel_dist=0.5):
    """Build one driver's telemetry entry."""
    return {"lap": lap, "dist": dist, "speed": speed, "rel_dist": rel_dist}


def frame(**drivers):
    """Build a frame from keyword-named drivers."""
    return {"t": 100.0, "drivers": drivers}


class TestFinishingOrder:
    def test_orders_by_laps_then_distance(self):
        rows = build_classification(frame(
            AAA=driver(58, 58 * LAP_DISTANCE - 500),
            BBB=driver(58, 58 * LAP_DISTANCE),
            CCC=driver(57, 57 * LAP_DISTANCE),
        ))
        assert [row["code"] for row in rows] == ["BBB", "AAA", "CCC"]
        assert rows[0]["gap_text"] == "WINNER"

    def test_retired_drivers_go_behind_every_finisher(self):
        # DDD covered more distance than the classified finisher CCC, but a
        # retirement is still classified behind anyone who took the flag.
        rows = build_classification(
            frame(AAA=driver(58, 58 * LAP_DISTANCE),
                  CCC=driver(55, 55 * LAP_DISTANCE),
                  DDD=driver(57, 57 * LAP_DISTANCE, rel_dist=1)),
            retirements={"DDD": {"lap": 57, "cause": "Engine"}},
        )
        assert [row["code"] for row in rows] == ["AAA", "CCC", "DDD"]
        assert rows[2]["status"] == "DNF"
        assert rows[2]["cause"] == "Engine"

    def test_rel_dist_marker_alone_detects_a_retirement(self):
        rows = build_classification(frame(
            AAA=driver(58, 58 * LAP_DISTANCE),
            BBB=driver(10, 10 * LAP_DISTANCE, rel_dist=1),
        ))
        assert rows[1]["status"] == "DNF"
        assert rows[1]["cause"] == "Retired"


class TestGaps:
    def test_time_gap_uses_the_leader_speed(self):
        # 500 units behind at 200 km/h -> 0.5 km / (200/3600 km/s) = 9 s
        rows = build_classification(frame(
            AAA=driver(58, 58 * LAP_DISTANCE),
            BBB=driver(58, 58 * LAP_DISTANCE - 500),
        ))
        assert rows[1]["gap"] == pytest.approx(9.0)
        assert rows[1]["gap_text"] == "+9.000s"

    @pytest.mark.parametrize("laps_behind, expected", [(1, "+1 LAP"), (2, "+2 LAPS")])
    def test_laps_down_is_singular_or_plural(self, laps_behind, expected):
        rows = build_classification(frame(
            AAA=driver(58, 58 * LAP_DISTANCE),
            BBB=driver(58 - laps_behind, (58 - laps_behind) * LAP_DISTANCE),
        ))
        assert rows[1]["gap_text"] == expected

    def test_a_zero_speed_leader_does_not_divide_by_zero(self):
        rows = build_classification(frame(
            AAA=driver(58, 58 * LAP_DISTANCE, speed=0.0),
            BBB=driver(58, 58 * LAP_DISTANCE - 500, speed=0.0),
        ))
        assert rows[1]["gap"] == pytest.approx(9.0)  # falls back to a default speed


class TestPoints:
    def test_top_ten_score_the_f1_points_table(self):
        rows = build_classification(frame(**{
            f"D{i:02d}": driver(58, 58 * LAP_DISTANCE - i) for i in range(12)
        }))
        assert [row["points"] for row in rows[:10]] == list(POINTS_BY_POSITION)

    def test_eleventh_and_below_score_nothing(self):
        rows = build_classification(frame(**{
            f"D{i:02d}": driver(58, 58 * LAP_DISTANCE - i) for i in range(12)
        }))
        assert rows[10]["points"] == 0
        assert rows[11]["points"] == 0

    def test_retirement_scores_nothing(self):
        rows = build_classification(
            frame(AAA=driver(58, 58 * LAP_DISTANCE),
                  BBB=driver(5, 5 * LAP_DISTANCE, rel_dist=1)),
            retirements={"BBB": {"lap": 5, "cause": "Gearbox"}},
        )
        assert rows[1]["points"] == 0

    def test_fastest_lap_adds_a_point_inside_the_top_ten(self):
        rows = build_classification(
            frame(AAA=driver(58, 58 * LAP_DISTANCE),
                  BBB=driver(58, 58 * LAP_DISTANCE - 10)),
            fastest_lap={"code": "BBB", "lap": 40, "time": 90.0},
        )
        assert rows[0]["points"] == 25
        assert rows[1]["points"] == 18 + 1
        assert rows[1]["fastest_lap"] is True

    def test_fastest_lap_outside_the_top_ten_scores_nothing(self):
        rows = build_classification(
            frame(**{f"D{i:02d}": driver(58, 58 * LAP_DISTANCE - i) for i in range(12)}),
            fastest_lap={"code": "D11", "lap": 40, "time": 90.0},
        )
        assert rows[11]["fastest_lap"] is True
        assert rows[11]["points"] == 0

    def test_a_retired_driver_never_takes_the_fastest_lap_point(self):
        rows = build_classification(
            frame(AAA=driver(58, 58 * LAP_DISTANCE),
                  BBB=driver(5, 5 * LAP_DISTANCE, rel_dist=1)),
            fastest_lap={"code": "BBB", "lap": 3, "time": 80.0},
        )
        assert rows[1]["points"] == 0


class TestNotClassified:
    def test_below_ninety_percent_distance_is_not_classified(self):
        rows = build_classification(frame(
            AAA=driver(58, 58 * LAP_DISTANCE),
            BBB=driver(40, 40 * LAP_DISTANCE),
        ))
        assert rows[1]["status"] == "Not classified"
        assert rows[1]["gap_text"] == "NC"
        assert rows[1]["points"] == 0

    def test_just_above_the_threshold_still_scores(self):
        rows = build_classification(frame(
            AAA=driver(58, 58 * LAP_DISTANCE),
            BBB=driver(55, 58 * LAP_DISTANCE * 0.95),
        ))
        assert rows[1]["status"] == "Finished"
        assert rows[1]["points"] == 18


class TestLapTimes:
    def _frames(self):
        # AAA runs 10s laps; BBB runs 12s then 16s.
        schedule = [(0, 1, 1), (10, 2, 1), (12, 2, 2), (20, 3, 2), (28, 3, 3)]
        return [
            {"t": float(t), "drivers": {
                "AAA": driver(a, a * LAP_DISTANCE),
                "BBB": driver(b, b * LAP_DISTANCE - 100),
            }}
            for t, a, b in schedule
        ]

    def test_lap_times_come_from_the_lap_counter_ticking_over(self):
        times = compute_lap_times(self._frames())
        assert times["AAA"] == [(1, 10.0), (2, 10.0)]
        assert times["BBB"] == [(1, 12.0), (2, 16.0)]

    def test_fastest_lap_is_the_quickest_measured_lap(self):
        fastest = find_fastest_lap(compute_lap_times(self._frames()))
        assert fastest == {"code": "AAA", "lap": 1, "time": 10.0}

    def test_no_measurable_laps_yields_no_fastest_lap(self):
        assert find_fastest_lap({}) is None
        assert find_fastest_lap(compute_lap_times([])) is None


class TestRobustness:
    @pytest.mark.parametrize("bad", [{}, {"drivers": {}}, {"t": 1.0, "drivers": {}}])
    def test_empty_frames_produce_no_rows(self, bad):
        assert build_classification(bad) == []

    def test_none_values_do_not_raise(self):
        rows = build_classification({"t": 0, "drivers": {
            "AAA": {"lap": None, "dist": None, "speed": None, "rel_dist": None},
        }})
        assert rows[0]["laps"] == 0

    def test_classification_from_no_frames(self):
        assert classification_from_frames([]) == {"classification": [],
                                                  "fastest_lap": None}


class TestRendering:
    def test_format_includes_names_and_the_fastest_lap(self):
        rows = build_classification(frame(AAA=driver(58, 58 * LAP_DISTANCE)))
        text = format_classification(
            rows, driver_names={"AAA": "Ann Aaa"},
            fastest_lap={"code": "AAA", "lap": 3, "time": 91.5},
        )
        assert "Ann Aaa" in text
        assert "91.5" in text

    def test_format_handles_an_empty_classification(self):
        assert "no results" in format_classification([])


class TestExport:
    def test_json_export_round_trips(self, tmp_path):
        rows = build_classification(frame(AAA=driver(58, 58 * LAP_DISTANCE)))
        path = tmp_path / "nested" / "results.json"

        assert export_classification(rows, str(path)) is True

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["classification"][0]["code"] == "AAA"

    def test_csv_export_writes_a_header(self, tmp_path):
        rows = build_classification(frame(AAA=driver(58, 58 * LAP_DISTANCE)))
        path = tmp_path / "results.csv"

        assert export_classification(rows, str(path)) is True
        assert path.read_text(encoding="utf-8").startswith("position,code")

    def test_nothing_to_export_reports_failure(self, tmp_path):
        assert export_classification([], str(tmp_path / "empty.json")) is False
