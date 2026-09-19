"""
Tests for the replay windows' frame interpolation.

The window classes need an OpenGL context to construct, so these tests call the
interpolation methods directly against a lightweight stand-in object. That is
deliberate: the maths is what matters here, and it keeps the suite runnable on a
headless machine.
"""

import types

import pytest

numpy = pytest.importorskip("numpy")
pytest.importorskip("arcade", reason="the replay windows need arcade")

import numpy as np  # noqa: E402

import src.arcade_replay as arcade_replay  # noqa: E402
import src.external_replay as external_replay  # noqa: E402
from src.f1_data import (  # noqa: E402
    FIELD_DIST, FIELD_DRS, FIELD_GEAR, FIELD_LAP, FIELD_POSITION,
    FIELD_REL_DIST, FIELD_SPEED, FIELD_TYRE, FIELD_X, FIELD_Y,
)

Window = arcade_replay.F1ReplayWindow


# --------------------------------------------------------------- NumPy path

N_FRAMES = 4
N_DRIVERS = 3
CODES = ["VER", "HAM", "LEC"]


def build_arrays():
    """Telemetry that changes by a known amount per frame, so lerps are checkable."""
    data = np.zeros((N_FRAMES, N_DRIVERS, 10), dtype=float)
    for frame in range(N_FRAMES):
        for driver in range(N_DRIVERS):
            data[frame, driver, FIELD_X] = 100 * driver + 10 * frame
            data[frame, driver, FIELD_Y] = 200 * driver + 20 * frame
            data[frame, driver, FIELD_DIST] = 1000 * frame + driver
            data[frame, driver, FIELD_REL_DIST] = 0.5
            data[frame, driver, FIELD_LAP] = frame + 1
            data[frame, driver, FIELD_TYRE] = 2
            data[frame, driver, FIELD_SPEED] = 100 + 10 * frame
            data[frame, driver, FIELD_GEAR] = 5
            data[frame, driver, FIELD_DRS] = 8
            data[frame, driver, FIELD_POSITION] = driver + 1

    metadata = np.array([[frame * 2.0, frame + 1] for frame in range(N_FRAMES)],
                        dtype=float)
    return data, metadata


@pytest.fixture
def numpy_window():
    """A stand-in carrying only what _get_frame_state() touches."""
    data, metadata = build_arrays()

    window = types.SimpleNamespace(
        use_numpy_arrays=True,
        driver_data_array=data,
        frame_metadata=metadata,
        driver_codes=CODES,
        n_frames=N_FRAMES,
        frames=None,
        _frame_state_cache_key=None,
        _frame_state_cache=None,
        _discrete_field_idx=np.array(arcade_replay.DISCRETE_FIELDS, dtype=np.intp),
    )
    window._build_state_from_arrays = lambda *a: Window._build_state_from_arrays(window, *a)
    window._build_state_from_frames = lambda *a: Window._build_state_from_frames(window, *a)
    return window


def state(window, frame_index):
    return Window._get_frame_state(window, frame_index)


class TestNumpyInterpolation:
    def test_an_exact_frame_is_returned_unblended(self, numpy_window):
        result = state(numpy_window, 1.0)
        assert result["drivers"]["VER"]["x"] == 10.0
        assert result["t"] == 2.0

    def test_continuous_fields_are_blended(self, numpy_window):
        driver = state(numpy_window, 1.5)["drivers"]["VER"]
        assert driver["x"] == 15.0        # 10 -> 20
        assert driver["y"] == 30.0        # 20 -> 40
        assert driver["dist"] == 1500.0   # 1000 -> 2000
        assert driver["speed"] == 115.0   # 110 -> 120

    def test_the_race_clock_is_blended(self, numpy_window):
        assert state(numpy_window, 1.5)["t"] == 3.0

    def test_discrete_fields_are_not_blended(self, numpy_window):
        driver = state(numpy_window, 1.5)["drivers"]["VER"]
        assert driver["lap"] == 2
        assert driver["tyre"] == 2
        assert driver["gear"] == 5
        assert driver["drs"] == 8
        assert driver["position"] == 1

    def test_the_retirement_marker_survives_interpolation(self, numpy_window):
        # rel_dist == 1 is how the replay knows a car is out. A blended value
        # would no longer be exactly 1 and the car would reappear.
        numpy_window.driver_data_array[1, 1, FIELD_REL_DIST] = 1.0
        numpy_window.driver_data_array[2, 1, FIELD_REL_DIST] = 1.0
        numpy_window._frame_state_cache_key = None

        assert state(numpy_window, 1.5)["drivers"]["HAM"]["rel_dist"] == 1

    def test_the_source_array_is_never_mutated(self, numpy_window):
        state(numpy_window, 1.5)
        assert numpy_window.driver_data_array[1, 0, FIELD_X] == 10.0
        assert numpy_window.driver_data_array[2, 0, FIELD_X] == 20.0

    def test_the_last_frame_has_nothing_to_blend_into(self, numpy_window):
        assert state(numpy_window, float(N_FRAMES - 1))["drivers"]["VER"]["x"] == 30.0

    @pytest.mark.parametrize("frame_index", [-5.0, 0.0, 99.0])
    def test_out_of_range_indices_are_clamped(self, numpy_window, frame_index):
        assert state(numpy_window, frame_index)["drivers"]["VER"]["x"] in (0.0, 30.0)

    def test_the_result_is_memoised_per_index(self, numpy_window):
        assert state(numpy_window, 2.25) is state(numpy_window, 2.25)


# -------------------------------------------------------------- legacy path

def legacy_frames():
    return [
        {"t": 0.0, "lap": 1, "drivers": {"VER": {
            "x": 0.0, "y": 0.0, "dist": 0.0, "speed": 100.0, "lap": 1,
            "tyre": 2, "rel_dist": 0.2, "gear": 5, "drs": 0, "position": 1}}},
        {"t": 1.0, "lap": 2, "drivers": {"VER": {
            "x": 10.0, "y": 20.0, "dist": 500.0, "speed": 200.0, "lap": 2,
            "tyre": 3, "rel_dist": 0.9, "gear": 7, "drs": 8, "position": 2}}},
    ]


@pytest.fixture
def legacy_window():
    window = types.SimpleNamespace(
        use_numpy_arrays=False, n_frames=2, frames=legacy_frames(),
        _frame_state_cache_key=None, _frame_state_cache=None,
    )
    window._build_state_from_arrays = lambda *a: Window._build_state_from_arrays(window, *a)
    window._build_state_from_frames = lambda *a: Window._build_state_from_frames(window, *a)
    return window


class TestLegacyInterpolation:
    def test_continuous_fields_are_blended(self, legacy_window):
        driver = state(legacy_window, 0.5)["drivers"]["VER"]
        assert (driver["x"], driver["y"]) == (5.0, 10.0)
        assert driver["dist"] == 250.0
        assert driver["speed"] == 150.0

    def test_discrete_fields_come_from_the_current_frame(self, legacy_window):
        driver = state(legacy_window, 0.5)["drivers"]["VER"]
        assert driver["lap"] == 1
        assert driver["tyre"] == 2
        assert driver["gear"] == 5
        assert driver["rel_dist"] == 0.2

    def test_the_stored_frame_is_not_mutated(self, legacy_window):
        state(legacy_window, 0.5)
        assert legacy_window.frames[0]["drivers"]["VER"]["x"] == 0.0

    def test_an_exact_frame_is_returned_as_is(self, legacy_window):
        assert state(legacy_window, 0.0) is legacy_window.frames[0]


# ------------------------------------------------- external replay window

ExternalWindow = external_replay.F1ReplayWindow


@pytest.fixture
def external_window():
    window = types.SimpleNamespace(
        frames=legacy_frames(), n_frames=2,
        _frame_state_cache_key=None, _frame_state_cache=None,
    )
    window.frames[0]["weather"] = {"air_temp": 20.0}
    window.frames[1]["weather"] = {"air_temp": 30.0}
    return window


class TestExternalInterpolation:
    def test_continuous_fields_are_blended(self, external_window):
        result = ExternalWindow._get_frame_state(external_window, 0.25)
        driver = result["drivers"]["VER"]
        assert (driver["x"], driver["y"]) == (2.5, 5.0)
        assert driver["speed"] == 125.0
        assert result["t"] == 0.25

    def test_weather_is_carried_across_unblended(self, external_window):
        result = ExternalWindow._get_frame_state(external_window, 0.25)
        assert result["weather"] == {"air_temp": 20.0}

    def test_discrete_fields_are_preserved(self, external_window):
        driver = ExternalWindow._get_frame_state(external_window, 0.25)["drivers"]["VER"]
        assert driver["tyre"] == 2
        assert driver["rel_dist"] == 0.2

    def test_the_result_is_memoised(self, external_window):
        first = ExternalWindow._get_frame_state(external_window, 0.5)
        assert first is ExternalWindow._get_frame_state(external_window, 0.5)


# ------------------------------------------- external replay track projection

@pytest.fixture
def projection_window():
    """A straight 1000 m reference line, so projected distances are obvious."""
    reference_x = np.linspace(0.0, 1000.0, 1001)
    reference_y = np.zeros_like(reference_x)
    cumulative = np.concatenate((
        [0.0], np.cumsum(np.hypot(np.diff(reference_x), np.diff(reference_y)))))

    window = types.SimpleNamespace(
        _ref_xs=reference_x, _ref_ys=reference_y, _ref_cumdist=cumulative,
        _ref_total_length=float(cumulative[-1]),
        _progress_cache_idx=None, _progress_cache={},
    )
    window._project_to_reference = lambda x, y: ExternalWindow._project_to_reference(
        window, x, y)
    return window


class TestTrackProjection:
    def _frame(self):
        return {"drivers": {
            "AAA": {"x": 250.5, "y": 3.0, "lap": 1},
            "BBB": {"x": 100.0, "y": 0.0, "lap": 3},
            "CCC": {"x": 0.0, "y": 0.0, "lap": 0},      # lap 0 is treated as lap 1
            "DDD": {"x": 500.0, "y": 0.0, "lap": None},  # defensive parse
        }}

    def test_progress_combines_completed_laps_with_the_projection(self, projection_window):
        progress = ExternalWindow._driver_progress(projection_window, 7, self._frame())
        assert progress["AAA"] == pytest.approx(250.5)
        assert progress["BBB"] == pytest.approx(2100.0)
        assert progress["CCC"] == pytest.approx(0.0)
        assert progress["DDD"] == pytest.approx(500.0)

    def test_the_same_stored_frame_reuses_the_cache(self, projection_window):
        first = ExternalWindow._driver_progress(projection_window, 7, self._frame())
        # A different frame body with the same index must not be recomputed.
        assert ExternalWindow._driver_progress(
            projection_window, 7, {"drivers": {}}) is first

    def test_a_new_frame_index_recomputes(self, projection_window):
        first = ExternalWindow._driver_progress(projection_window, 7, self._frame())
        second = ExternalWindow._driver_progress(projection_window, 8, self._frame())
        assert second is not first
        assert second == first

    def test_a_degenerate_reference_line_does_not_raise(self):
        window = types.SimpleNamespace(
            _ref_xs=np.array([]), _ref_ys=np.array([]),
            _ref_cumdist=np.array([0.0]), _ref_total_length=0.0,
            _progress_cache_idx=None, _progress_cache={},
        )
        window._project_to_reference = lambda x, y: ExternalWindow._project_to_reference(
            window, x, y)

        progress = ExternalWindow._driver_progress(
            window, 0, {"drivers": {"AAA": {"x": 1.0, "y": 2.0, "lap": 2}}})
        assert progress == {"AAA": 0.0}
