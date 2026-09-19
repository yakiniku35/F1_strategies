"""Tests for the race simulator's retirement (DNF) modelling."""

import random
import statistics

import pytest

from src.simulation.race_simulator import PredictedRaceSimulator

DRIVERS = [f"D{i:02d}" for i in range(20)]
VALID_CAUSES = {cause for cause, _ in PredictedRaceSimulator.DNF_CAUSES}


class _Knobs:
    """Only the class-level knobs are needed to drive _generate_retirements."""

    DNF_PROBABILITY = PredictedRaceSimulator.DNF_PROBABILITY
    DNF_CAUSES = PredictedRaceSimulator.DNF_CAUSES
    FIRST_LAP_CAUSES = PredictedRaceSimulator.FIRST_LAP_CAUSES
    FIRST_LAP_DNF_CHANCE = PredictedRaceSimulator.FIRST_LAP_DNF_CHANCE
    DNF_LAP_PEAK = PredictedRaceSimulator.DNF_LAP_PEAK


def generate(total_laps=58, drivers=None, knobs=None, seed=None):
    """Call the generator unbound, with a controllable RNG seed."""
    if seed is not None:
        random.seed(seed)
    return PredictedRaceSimulator._generate_retirements(
        knobs or _Knobs(), total_laps, DRIVERS if drivers is None else drivers
    )


class TestShape:
    def test_every_retirement_is_a_known_driver_with_a_valid_lap_and_cause(self):
        retirements = generate(seed=1)

        assert set(retirements) <= set(DRIVERS)
        for info in retirements.values():
            assert 1 <= info["lap"] <= 58
            assert info["cause"] in VALID_CAUSES

    def test_the_same_seed_gives_the_same_race(self):
        assert generate(seed=42) == generate(seed=42)

    def test_different_seeds_eventually_differ(self):
        assert any(generate(seed=s) != generate(seed=0) for s in range(1, 20))


class TestDistribution:
    def test_the_average_retirement_count_is_realistic(self):
        counts = [len(generate(seed=seed)) for seed in range(400)]
        # Roughly 10% of a twenty-car field: two to three per race.
        assert 1.4 <= statistics.mean(counts) <= 2.6
        assert 0 <= min(counts) and max(counts) <= 20

    def test_retirements_cluster_in_the_first_half(self):
        laps = [info["lap"]
                for seed in range(400)
                for info in generate(seed=seed).values()]
        first_half = sum(1 for lap in laps if lap <= 29) / len(laps)
        assert first_half > 0.6


class TestEdgeCases:
    def test_no_drivers_means_no_retirements(self):
        assert generate(drivers=[], seed=0) == {}

    def test_a_one_lap_race_retires_on_lap_one(self):
        for info in generate(total_laps=1, seed=0).values():
            assert info["lap"] == 1

    @pytest.mark.parametrize("total_laps", [0, -5])
    def test_a_nonsense_lap_count_still_yields_a_valid_lap(self, total_laps):
        for info in generate(total_laps=total_laps, seed=0).values():
            assert info["lap"] >= 1

    def test_a_certain_dnf_rate_retires_everyone(self):
        class AllOut(_Knobs):
            DNF_PROBABILITY = 1.0

        assert len(generate(knobs=AllOut(), seed=7)) == len(DRIVERS)

    def test_a_zero_dnf_rate_retires_nobody(self):
        class NoneOut(_Knobs):
            DNF_PROBABILITY = 0.0

        assert generate(knobs=NoneOut(), seed=7) == {}
