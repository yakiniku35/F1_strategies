"""Tests for tyre degradation and the strategy heuristics."""

import pytest

from src.lib.tyres import get_tyre_compound_int, get_tyre_compound_str
from src.simulation.race_dynamics import TyreDegradation


class TestTyreCompoundCodes:
    @pytest.mark.parametrize("name, code", [
        ("SOFT", 0), ("MEDIUM", 1), ("HARD", 2), ("INTERMEDIATE", 3), ("WET", 4),
    ])
    def test_names_and_codes_round_trip(self, name, code):
        assert get_tyre_compound_int(name) == code
        assert get_tyre_compound_str(code) == name

    def test_lowercase_names_are_accepted(self):
        assert get_tyre_compound_int("soft") == 0

    def test_unknown_values_are_reported_rather_than_raising(self):
        assert get_tyre_compound_int("BANANA") == -1
        assert get_tyre_compound_str(99) == "UNKNOWN"


class TestTyreDegradation:
    def test_fresh_tyres_sit_at_their_peak(self):
        degradation = TyreDegradation()
        degradation.initialize_driver("VER", "SOFT")

        expected = TyreDegradation.COMPOUNDS["SOFT"]["peak_performance"]
        assert degradation.get_performance_factor("VER") == pytest.approx(expected)

    def test_performance_falls_as_the_tyres_age(self):
        degradation = TyreDegradation()
        degradation.initialize_driver("VER", "SOFT")

        fresh = degradation.get_performance_factor("VER")
        degradation.age_tyres("VER", 5)
        assert degradation.get_performance_factor("VER") < fresh

    def test_degradation_never_falls_below_the_floor(self):
        degradation = TyreDegradation()
        degradation.initialize_driver("VER", "SOFT")
        degradation.age_tyres("VER", 500)

        floor = TyreDegradation.COMPOUNDS["SOFT"]["min_performance"]
        assert degradation.get_performance_factor("VER") == pytest.approx(floor)

    def test_the_cliff_makes_performance_drop_faster(self):
        spec = TyreDegradation.COMPOUNDS["MEDIUM"]

        before = TyreDegradation()
        before.initialize_driver("VER", "MEDIUM")
        before.age_tyres("VER", spec["cliff_lap"] - 1)

        after = TyreDegradation()
        after.initialize_driver("VER", "MEDIUM")
        after.age_tyres("VER", spec["cliff_lap"])

        drop_at_cliff = (before.get_performance_factor("VER")
                         - after.get_performance_factor("VER"))
        assert drop_at_cliff > spec["degradation_rate"]

    def test_softer_compounds_start_quicker_and_wear_faster(self):
        rates = {name: TyreDegradation.COMPOUNDS[name]["degradation_rate"]
                 for name in ("SOFT", "MEDIUM", "HARD")}
        peaks = {name: TyreDegradation.COMPOUNDS[name]["peak_performance"]
                 for name in ("SOFT", "MEDIUM", "HARD")}

        assert peaks["SOFT"] > peaks["MEDIUM"] > peaks["HARD"]
        assert rates["SOFT"] > rates["MEDIUM"] > rates["HARD"]

    def test_changing_tyres_resets_the_age(self):
        degradation = TyreDegradation()
        degradation.initialize_driver("VER", "SOFT")
        degradation.age_tyres("VER", 12)

        degradation.change_tyres("VER", "HARD")

        assert degradation.tyre_ages["VER"] == 0
        expected = TyreDegradation.COMPOUNDS["HARD"]["peak_performance"]
        assert degradation.get_performance_factor("VER") == pytest.approx(expected)

    def test_an_unknown_driver_is_neutral_rather_than_an_error(self):
        assert TyreDegradation().get_performance_factor("NOBODY") == 1.0

    def test_ageing_an_unknown_driver_is_a_no_op(self):
        degradation = TyreDegradation()
        degradation.age_tyres("NOBODY", 5)  # must not raise
        assert "NOBODY" not in degradation.tyre_ages
