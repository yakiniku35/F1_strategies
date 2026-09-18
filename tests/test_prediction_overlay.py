"""Tests for the prediction overlay's cached battle detection."""

import itertools
import random

import pytest

pytest.importorskip("arcade", reason="the overlay needs arcade")

from src.dashboard.prediction_overlay import PredictionOverlay  # noqa: E402


def brute_force_battles(predictions):
    """The original all-pairs definition, used here as the reference answer."""
    battling = set()

    for left, right in itertools.permutations(predictions, 2):
        left_position = predictions[left].get("predicted_position")
        right_position = predictions[right].get("predicted_position")
        if left_position is None or right_position is None:
            continue
        if abs(left_position - right_position) < 0.5:
            battling.add(left)

    return battling


class TestBattleCache:
    def test_matches_the_all_pairs_definition_on_random_inputs(self):
        random.seed(0)

        for _ in range(300):
            predictions = {
                f"D{i:02d}": {"predicted_position": random.choice(
                    [None, round(random.uniform(1, 12), 2)])}
                for i in range(random.randint(0, 12))
            }

            overlay = PredictionOverlay(predictions)
            found = {code for code in predictions if overlay.is_in_battle(code)}

            assert found == brute_force_battles(predictions)

    def test_drivers_far_apart_are_not_battling(self):
        overlay = PredictionOverlay({"A": {"predicted_position": 1.0},
                                     "B": {"predicted_position": 9.0}})
        assert not overlay.is_in_battle("A")
        assert not overlay.is_in_battle("B")

    def test_drivers_close_together_are_battling(self):
        overlay = PredictionOverlay({"A": {"predicted_position": 1.0},
                                     "B": {"predicted_position": 1.2}})
        assert overlay.is_in_battle("A")
        assert overlay.is_in_battle("B")

    def test_updating_predictions_refreshes_the_cache(self):
        overlay = PredictionOverlay({"A": {"predicted_position": 1.0},
                                     "B": {"predicted_position": 9.0}})
        assert not overlay.is_in_battle("A")

        overlay.update_predictions({"A": {"predicted_position": 1.0},
                                    "B": {"predicted_position": 1.2}})
        assert overlay.is_in_battle("A")

        overlay.update_predictions({})
        assert not overlay.is_in_battle("A")

    def test_missing_predictions_are_handled(self):
        assert not PredictionOverlay().is_in_battle("A")
        assert not PredictionOverlay({"A": {}}).is_in_battle("A")
        assert not PredictionOverlay(
            {"A": {"predicted_position": None}}).is_in_battle("A")

    def test_an_unknown_driver_is_never_battling(self):
        overlay = PredictionOverlay({"A": {"predicted_position": 1.0},
                                     "B": {"predicted_position": 1.1}})
        assert not overlay.is_in_battle("ZZZ")
