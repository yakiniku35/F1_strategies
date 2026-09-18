"""Tests for the season-aware schedule and driver lineup provider."""

import sys
import types
from datetime import datetime

import pandas as pd
import pytest

from src.simulation.future_race_data import FutureRaceDataProvider


@pytest.fixture
def fake_fastf1(monkeypatch):
    """Install a controllable stand-in for FastF1 and hand it to the test."""
    module = types.ModuleType("fastf1")

    module.get_event_schedule = lambda year, include_testing=False: pd.DataFrame([
        {"RoundNumber": 1, "EventName": "Bahrain Grand Prix",
         "Location": "Sakhir", "EventDate": "2026-03-08"},
        {"RoundNumber": 2, "EventName": "Neverland Grand Prix",
         "Location": "Neverland", "EventDate": "2026-03-22"},
    ])

    class _Session:
        results = pd.DataFrame([
            {"Abbreviation": "AAA", "FullName": "Ann Aaa",
             "TeamName": "Team One", "DriverNumber": 1},
            {"Abbreviation": "BBB", "FullName": "Bo Bbb",
             "TeamName": "Team Two", "DriverNumber": 2},
        ])

        def load(self, **kwargs):
            pass

    module.get_session = lambda year, rnd, kind: _Session()

    monkeypatch.setitem(sys.modules, "fastf1", module)
    return module


class TestFallback:
    def test_the_bundled_year_uses_the_bundled_table(self):
        provider = FutureRaceDataProvider(FutureRaceDataProvider.FALLBACK_YEAR)
        assert provider.schedule_source == "fallback"
        assert provider.drivers_source == "fallback"
        assert len(provider.get_schedule()) == len(provider.FALLBACK_SCHEDULE)

    def test_defaults_to_the_current_calendar_year(self):
        assert FutureRaceDataProvider().year == datetime.now().year

    def test_returned_data_is_a_copy(self):
        provider = FutureRaceDataProvider(2025)

        provider.get_schedule()[0]["laps"] = -1
        provider.get_drivers_list()[0]["team"] = "Mutated"

        assert FutureRaceDataProvider.FALLBACK_SCHEDULE[0]["laps"] != -1
        assert FutureRaceDataProvider.FALLBACK_DRIVERS[0]["team"] != "Mutated"

    def test_lookups_work_against_the_bundled_calendar(self):
        provider = FutureRaceDataProvider(2025)
        assert provider.get_race_by_name("Monaco")["laps"] == 78
        assert provider.get_race_by_round(1)["round"] == 1
        assert provider.get_race_by_name("Nowhere-at-all") is None
        assert provider.get_race_by_round(999) is None

    def test_source_is_resolved_even_when_read_first(self):
        # Reading the source before the data must not report a misleading None.
        assert FutureRaceDataProvider(2025).schedule_source == "fallback"
        assert FutureRaceDataProvider(2025).drivers_source == "fallback"


class TestLiveSchedule:
    def test_a_non_bundled_year_comes_from_fastf1(self, fake_fastf1):
        provider = FutureRaceDataProvider(2026)
        schedule = provider.get_schedule()

        assert provider.schedule_source == "fastf1"
        assert [race["round"] for race in schedule] == [1, 2]
        assert schedule[0]["gp"] == "Bahrain"
        assert schedule[0]["date"] == "2026-03-08"

    def test_lap_counts_are_reused_from_the_bundled_table(self, fake_fastf1):
        schedule = FutureRaceDataProvider(2026).get_schedule()
        assert schedule[0]["laps"] == 57   # Bahrain, known circuit
        assert schedule[1]["laps"] == 55   # unknown circuit -> neutral default

    def test_the_driver_lineup_comes_from_the_latest_results(self, fake_fastf1):
        provider = FutureRaceDataProvider(2026)

        assert [d["code"] for d in provider.get_drivers_list()] == ["AAA", "BBB"]
        assert provider.drivers_source == "fastf1"
        assert provider.get_driver_by_code("AAA")["team"] == "Team One"

    def test_standings_are_derived_from_the_live_lineup(self, fake_fastf1):
        standings = FutureRaceDataProvider(2026).get_current_standings()
        assert {row["code"] for row in standings} == {"AAA", "BBB"}


class TestFailureHandling:
    def test_a_raising_fastf1_falls_back(self, fake_fastf1):
        def boom(*args, **kwargs):
            raise RuntimeError("no network")

        fake_fastf1.get_event_schedule = boom
        fake_fastf1.get_session = boom

        provider = FutureRaceDataProvider(2027)
        assert provider.schedule_source == "fallback"
        assert provider.drivers_source == "fallback"
        assert provider.get_schedule()

    def test_an_empty_fastf1_schedule_falls_back(self, fake_fastf1):
        fake_fastf1.get_event_schedule = lambda year, include_testing=False: pd.DataFrame([])

        provider = FutureRaceDataProvider(2028)
        assert provider.schedule_source == "fallback"
        assert provider.get_schedule()

    def test_sessions_without_results_fall_back_for_drivers(self, fake_fastf1):
        class _Empty:
            results = pd.DataFrame([])

            def load(self, **kwargs):
                pass

        fake_fastf1.get_session = lambda year, rnd, kind: _Empty()

        provider = FutureRaceDataProvider(2029)
        assert provider.drivers_source == "fallback"
        assert provider.get_drivers_list()

    def test_an_unparseable_race_date_does_not_raise(self, fake_fastf1):
        fake_fastf1.get_event_schedule = lambda year, include_testing=False: pd.DataFrame([
            {"RoundNumber": 1, "EventName": "Broken Grand Prix",
             "Location": "Nowhere", "EventDate": "not-a-date"},
        ])

        provider = FutureRaceDataProvider(2030)
        assert provider.get_schedule()[0]["date"].startswith("2030")
        assert provider.is_future_race(2030, "Broken") is True
