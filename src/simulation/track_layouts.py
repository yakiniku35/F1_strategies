"""
Track Layout Manager Module.
Manages track layouts for race simulation visualization.
"""

import os
import json
import numpy as np
from typing import Optional, Tuple
import fastf1


class TrackLayoutManager:
    """
    Manages track layouts for F1 circuits.
    Provides track geometry data for visualization.
    """

    # Layout constants
    DEFAULT_TRACK_WIDTH = 150
    OVAL_TRACK_POINTS = 100

    # Pre-defined simplified track layouts (fallback when no historical data)
    # Format: list of (x, y) points representing the track centerline
    SIMPLIFIED_LAYOUTS = {
        "Monaco": {
            "center_x": [0, 100, 200, 250, 200, 100, 50, 100, 200, 250, 200, 100, 0],
            "center_y": [0, 50, 50, 100, 150, 150, 200, 250, 250, 200, 150, 100, 0],
            "track_width": 150,
        },
        "Monza": {
            "center_x": [0, 200, 400, 500, 500, 400, 200, 0, -100, -100, 0],
            "center_y": [0, 50, 100, 200, 350, 400, 400, 350, 200, 100, 0],
            "track_width": 200,
        },
        "Silverstone": {
            "center_x": [0, 150, 300, 400, 400, 300, 200, 100, -50, -100, 0],
            "center_y": [0, -50, 0, 100, 250, 350, 400, 350, 250, 100, 0],
            "track_width": 180,
        },
    }

    def __init__(self, cache_dir: str = ".fastf1-cache"):
        """
        Initialize the track layout manager.

        Args:
            cache_dir: Directory for FastF1 cache
        """
        self.cache_dir = cache_dir
        self._cached_layouts = {}

        # Ensure cache directory exists
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def get_track_layout(self, gp_name: str, year: int = 2024) -> dict:
        """
        Get track layout for a Grand Prix.
        Tries to load from historical data first, falls back to simplified layout.

        Args:
            gp_name: Grand Prix name
            year: Year to fetch layout from (defaults to recent year)

        Returns:
            Dictionary with track layout data
        """
        cache_key = f"{gp_name}_{year}"

        if cache_key in self._cached_layouts:
            return self._cached_layouts[cache_key]

        # Try to load from historical data
        layout = self._load_from_historical(year, gp_name)

        if layout is None:
            # Try previous year
            layout = self._load_from_historical(year - 1, gp_name)

        if layout is None:
            # Fall back to simplified layout
            layout = self._get_simplified_layout(gp_name)

        self._cached_layouts[cache_key] = layout
        return layout

    def _load_from_historical(self, year: int, gp_name: str) -> Optional[dict]:
        """
        Load track layout from historical FastF1 data.

        Args:
            year: Year of the race
            gp_name: Grand Prix name

        Returns:
            Layout dictionary or None if not available
        """
        try:
            # Enable cache
            fastf1.Cache.enable_cache(self.cache_dir)

            # Load session
            session = fastf1.get_session(year, gp_name, 'R')
            session.load(telemetry=True)

            # Get fastest lap for track layout
            try:
                fastest_lap = session.laps.pick_fastest()
                telemetry = fastest_lap.get_telemetry()
            except (IndexError, AttributeError):
                # Try first available lap
                if len(session.laps) > 0:
                    telemetry = session.laps.iloc[0].get_telemetry()
                else:
                    return None

            if telemetry.empty:
                return None

            # Extract X, Y coordinates
            x_coords = telemetry["X"].to_numpy()
            y_coords = telemetry["Y"].to_numpy()

            # Calculate track bounds
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            return {
                "x": x_coords.tolist(),
                "y": y_coords.tolist(),
                "track_width": self.DEFAULT_TRACK_WIDTH,
                "x_min": float(x_min),
                "x_max": float(x_max),
                "y_min": float(y_min),
                "y_max": float(y_max),
                "source": "historical",
                "year": year,
                "gp": gp_name,
            }

        except Exception as e:
            print(f"Could not load historical layout for {year} {gp_name}: {e}")
            return None

    def _get_simplified_layout(self, gp_name: str) -> dict:
        """
        Get a simplified/fallback track layout.

        Args:
            gp_name: Grand Prix name

        Returns:
            Simplified layout dictionary
        """
        # Try to find matching simplified layout
        gp_lower = gp_name.lower()
        for track_name, layout in self.SIMPLIFIED_LAYOUTS.items():
            if track_name.lower() in gp_lower or gp_lower in track_name.lower():
                x = layout["center_x"]
                y = layout["center_y"]
                return {
                    "x": x,
                    "y": y,
                    "track_width": layout["track_width"],
                    "x_min": min(x),
                    "x_max": max(x),
                    "y_min": min(y),
                    "y_max": max(y),
                    "source": "simplified",
                    "gp": gp_name,
                }

        # Generate generic oval track
        return self._generate_oval_track(gp_name)

    def _generate_oval_track(self, gp_name: str) -> dict:
        """
        Generate a generic oval track layout.

        Args:
            gp_name: Grand Prix name

        Returns:
            Generic oval layout dictionary
        """
        # Generate oval points
        t = np.linspace(0, 2 * np.pi, self.OVAL_TRACK_POINTS)
        x = (300 * np.cos(t)).tolist()
        y = (200 * np.sin(t)).tolist()

        return {
            "x": x,
            "y": y,
            "track_width": self.DEFAULT_TRACK_WIDTH,
            "x_min": -300,
            "x_max": 300,
            "y_min": -200,
            "y_max": 200,
            "source": "generated",
            "gp": gp_name,
        }

    def get_track_geometry(self, layout: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate track geometry (inner and outer edges) from centerline.

        Args:
            layout: Track layout dictionary

        Returns:
            Tuple of (x_inner, y_inner, x_outer, y_outer) numpy arrays
        """
        x = np.array(layout["x"])
        y = np.array(layout["y"])
        track_width = layout.get("track_width", 150) / 2

        # Compute tangents
        dx = np.gradient(x)
        dy = np.gradient(y)

        # Normalize
        norm = np.sqrt(dx**2 + dy**2)
        norm[norm == 0] = 1.0
        dx /= norm
        dy /= norm

        # Normal vectors
        nx = -dy
        ny = dx

        # Outer and inner edges
        x_outer = x + nx * track_width
        y_outer = y + ny * track_width
        x_inner = x - nx * track_width
        y_inner = y - ny * track_width

        return x_inner, y_inner, x_outer, y_outer

    def save_layout_to_file(self, layout: dict, filepath: str):
        """
        Save a track layout to a JSON file.

        Args:
            layout: Track layout dictionary
            filepath: Path to save the file
        """
        with open(filepath, 'w') as f:
            json.dump(layout, f, indent=2)

    def load_layout_from_file(self, filepath: str) -> Optional[dict]:
        """
        Load a track layout from a JSON file.

        Args:
            filepath: Path to the layout file

        Returns:
            Layout dictionary or None if file not found
        """
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def interpolate_position(self, layout: dict, progress: float) -> Tuple[float, float]:
        """
        Interpolate a position on the track given progress (0-1).

        Args:
            layout: Track layout dictionary
            progress: Progress around the track (0.0 to 1.0)

        Returns:
            Tuple of (x, y) coordinates
        """
        x = np.array(layout["x"])
        y = np.array(layout["y"])

        # Ensure progress is in [0, 1]
        progress = progress % 1.0

        # Find interpolated position
        idx = progress * (len(x) - 1)
        idx_low = int(idx)
        idx_high = min(idx_low + 1, len(x) - 1)
        frac = idx - idx_low

        x_pos = x[idx_low] * (1 - frac) + x[idx_high] * frac
        y_pos = y[idx_low] * (1 - frac) + y[idx_high] * frac

        return float(x_pos), float(y_pos)

    def create_example_lap_dataframe(self, layout: dict):
        """
        Create a pandas DataFrame similar to FastF1 telemetry for arcade_replay.

        Args:
            layout: Track layout dictionary

        Returns:
            DataFrame with X, Y columns
        """
        import pandas as pd

        return pd.DataFrame({
            "X": layout["x"],
            "Y": layout["y"],
        })
