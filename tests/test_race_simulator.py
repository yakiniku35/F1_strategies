"""Smoke tests for the prediction flow that ``main.py --predict`` runs.

2025 is the bundled fallback season, so these use no network: the provider
never asks FastF1 for the calendar, and the track layout falls back to the
built-in shapes when the historical lookup fails.
"""

import pytest

from src.simulation.race_simulator import PredictedRaceSimulator


@pytest.fixture
def simulator():
    return PredictedRaceSimulator(2025, "Monaco")


def test_qualifying_prediction_ranks_the_whole_grid(simulator):
    # estimate_qualifying() used to read the lazily filled points table
    # directly and crash on None before anything had populated it.
    qualifying = simulator.get_qualifying_results()

    drivers = simulator.data_provider.get_drivers_list()
    assert len(qualifying) == len(drivers)
    assert [q["grid"] for q in qualifying] == list(range(1, len(drivers) + 1))
    assert {q["code"] for q in qualifying} == {d["code"] for d in drivers}


def test_prediction_confidence_covers_every_driver(simulator):
    confidences = simulator.get_prediction_confidence()

    codes = {d["code"] for d in simulator.data_provider.get_drivers_list()}
    assert codes <= set(confidences)
    assert all(0 <= value <= 1 for value in confidences.values())


def test_team_colors_follow_the_loaded_lineup(simulator):
    # This used to read a DRIVERS_2025 attribute the provider no longer has.
    colors = simulator._get_team_colors()

    codes = {d["code"] for d in simulator.data_provider.get_drivers_list()}
    assert set(colors) == codes
    for rgb in colors.values():
        assert len(rgb) == 3
        assert all(0 <= channel <= 255 for channel in rgb)


def test_simulated_frames_can_open_the_replay_window(simulator):
    # Everything run_arcade_replay() is handed by predict_future_race(), which
    # simulates the full race distance. (Very short races are not supported:
    # incident generation assumes at least ten laps.)
    sim_data = simulator.generate_simulated_frames()

    assert sim_data["frames"]
    assert sim_data["total_laps"] == simulator.race_info["laps"]
    assert set(sim_data["driver_colors"]) == set(sim_data["drivers"])
    for key in ("track_statuses", "example_lap", "total_laps"):
        assert key in sim_data
