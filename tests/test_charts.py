"""Tests for the race chart module."""

import pytest

matplotlib = pytest.importorskip("matplotlib", reason="charts need matplotlib")

from src.dashboard.charts import (  # noqa: E402
    build_stints,
    extract_lap_history,
    generate_race_charts,
    plot_position_changes,
    plot_tyre_strategy,
)


def build_frames(total_laps=20, retirements=None, frames_per_lap=3):
    """Synthetic telemetry for three drivers, with optional retirements."""
    retirements = retirements or {}
    frames = []

    for lap in range(1, total_laps + 1):
        for sub in range(frames_per_lap):
            drivers = {}
            for index, code in enumerate(["AAA", "BBB", "CCC"]):
                out = code in retirements and lap >= retirements[code]
                drivers[code] = {
                    "lap": lap,
                    "position": index + 1,
                    "dist": lap * 5000.0,
                    "rel_dist": 1 if out else 0.5,
                    "tyre": 0 if lap <= 10 else 2,   # SOFT then HARD
                }
            frames.append({"t": float(lap * 3 + sub), "lap": lap, "drivers": drivers})

    return frames


class TestStints:
    def test_no_data_means_no_stints(self):
        assert build_stints({}) == []

    def test_consecutive_laps_on_one_compound_form_a_single_stint(self):
        stints = build_stints({1: "SOFT", 2: "SOFT", 3: "MEDIUM", 4: "MEDIUM", 5: "MEDIUM"})
        assert [(s["compound"], s["start_lap"], s["end_lap"], s["laps"]) for s in stints] == [
            ("SOFT", 1, 2, 2), ("MEDIUM", 3, 5, 3),
        ]

    def test_a_gap_in_the_laps_starts_a_new_stint(self):
        stints = build_stints({1: "SOFT", 2: "SOFT", 9: "SOFT"})
        assert len(stints) == 2
        assert stints[1]["start_lap"] == 9

    def test_refitting_the_same_compound_is_its_own_stint(self):
        stints = build_stints({1: "SOFT", 2: "HARD", 3: "SOFT"})
        assert [s["compound"] for s in stints] == ["SOFT", "HARD", "SOFT"]


class TestHistory:
    def test_one_record_per_driver_per_lap(self):
        history = extract_lap_history(build_frames())
        assert history["laps"] == list(range(1, 21))
        assert set(history["drivers"]) == {"AAA", "BBB", "CCC"}

    def test_a_retired_driver_stops_being_recorded(self):
        history = extract_lap_history(build_frames(retirements={"CCC": 8}))
        assert max(history["positions"]["CCC"]) == 7
        assert max(history["positions"]["AAA"]) == 20
        assert history["retired"] == {"CCC"}

    def test_integer_compound_codes_are_resolved(self):
        history = extract_lap_history(build_frames())
        assert history["compounds"]["AAA"][5] == "SOFT"
        assert history["compounds"]["AAA"][15] == "HARD"

    def test_string_compounds_are_accepted_too(self):
        history = extract_lap_history([{"t": 0, "lap": 1, "drivers": {
            "AAA": {"lap": 1, "position": 1, "tyre": "medium", "rel_dist": 0.2},
        }}])
        assert history["compounds"]["AAA"][1] == "MEDIUM"

    def test_drivers_are_ordered_by_final_position(self):
        history = extract_lap_history(build_frames())
        assert history["drivers"][0] == "AAA"

    @pytest.mark.parametrize("frames", [
        [],
        [{"drivers": {"A": {"lap": None, "position": None}}}],
        [{"drivers": {"A": {"lap": 1, "position": 0}}}],
    ])
    def test_unusable_input_yields_no_drivers(self, frames):
        assert extract_lap_history(frames)["drivers"] == []


class TestRendering:
    @pytest.mark.parametrize("theme", ["light", "dark"])
    def test_both_charts_render_in_both_themes(self, tmp_path, theme):
        charts = generate_race_charts(
            build_frames(retirements={"CCC": 8}),
            driver_colors={"AAA": (30, 65, 255), "BBB": (30, 65, 255), "CCC": (220, 0, 0)},
            output_dir=str(tmp_path / theme),
            title_prefix="2026 Testing GP",
            theme=theme,
        )

        assert set(charts) == {"positions", "tyres"}
        for path in charts.values():
            assert path is not None
            written = tmp_path / theme
            assert (written / path.split("/")[-1]).stat().st_size > 8000

    def test_a_full_twenty_car_field_renders(self, tmp_path):
        codes = [f"D{i:02d}" for i in range(20)]
        frames = [
            {"t": float(lap), "lap": lap, "drivers": {
                code: {"lap": lap, "position": index + 1, "rel_dist": 0.3,
                       "tyre": 0 if lap < 20 else 2}
                for index, code in enumerate(codes)
            }}
            for lap in range(1, 59)
        ]

        charts = generate_race_charts(frames, output_dir=str(tmp_path))
        assert all(charts.values())

    def test_empty_history_writes_nothing(self, tmp_path):
        history = extract_lap_history([])
        assert plot_position_changes(history, output_path=str(tmp_path / "p.png")) is None
        assert plot_tyre_strategy(history, output_path=str(tmp_path / "t.png")) is None

    def test_missing_team_colours_still_render(self, tmp_path):
        charts = generate_race_charts(build_frames(), driver_colors=None,
                                      output_dir=str(tmp_path))
        assert all(charts.values())
