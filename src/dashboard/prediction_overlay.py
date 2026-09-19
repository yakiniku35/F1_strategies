"""
F1 Prediction Overlay Module.
Visualizes ML prediction results on the race simulation.
"""

import arcade
from typing import Optional

# Tables view layout constants
TABLES_BACKGROUND_WIDTH_RATIO = 0.8
TABLES_BACKGROUND_HEIGHT_RATIO = 0.8


class PredictionOverlay:
    """
    Overlay for displaying ML predictions on the race simulation.

    Provides visual indicators for:
    - Predicted position trends (arrows)
    - Predicted position changes
    - Pit stop suggestion windows
    - Battle predictions highlighting
    """

    def __init__(self, predictions: Optional[dict] = None):
        """
        Initialize the prediction overlay.

        Args:
            predictions: Dictionary of predictions keyed by driver code
        """
        self.predictions = predictions or {}
        self.show_overlay = True
        self.show_tables = False
        # Set of drivers currently in a predicted battle. Derived from
        # self.predictions and refreshed by _rebuild_battle_cache().
        self._battle_drivers = set()
        self._rebuild_battle_cache()
        # Long-lived arcade.Text objects keyed by call site (see _get_text()).
        self._text_cache = {}
        self._text_colors = {}

    def _get_text(self, key, text, x, y, color, font_size=12, bold=False,
                  anchor_x="left", anchor_y="baseline"):
        """Return a reusable arcade.Text object for one call site.

        The tables view alone draws about eighty labels. Rebuilding an
        arcade.Text lays the string out glyph by glyph and uploads it to the GPU,
        so doing that every frame is what made the tables view drop frames. Each
        call site now keeps one object and only changed attributes are written.

        Args:
            key: Unique, stable identifier for the call site.
            text: The string to display.
            x, y: Screen position.
            color: Text colour as an RGB or RGBA tuple.
            font_size, bold, anchor_x, anchor_y: Applied once, at construction.

        Returns:
            The cached arcade.Text, ready to ``.draw()``.
        """
        label = self._text_cache.get(key)

        if label is None:
            label = arcade.Text(text, x, y, color, font_size, bold=bold,
                                anchor_x=anchor_x, anchor_y=anchor_y)
            self._text_cache[key] = label
            self._text_colors[key] = color
            return label

        if label.text != text:
            label.text = text
        if label.x != x:
            label.x = x
        if label.y != y:
            label.y = y
        if self._text_colors[key] != color:
            label.color = color
            self._text_colors[key] = color

        return label

    def update_predictions(self, predictions: dict):
        """
        Update the predictions data.

        Args:
            predictions: New predictions dictionary
        """
        self.predictions = predictions
        self._rebuild_battle_cache()

    def _rebuild_battle_cache(self):
        """Recompute which drivers are in a predicted battle.

        Deciding this compares every driver against every other one, so doing it
        while drawing cost O(drivers^2) on each of the ~60 frames rendered every
        second. The answer only changes when the predictions change, so it is
        computed here instead - once per prediction refresh.

        Sorting the predicted positions turns the all-pairs comparison into a
        single walk over neighbouring entries.
        """
        ranked = sorted(
            (pred.get('predicted_position'), code)
            for code, pred in self.predictions.items()
            if pred.get('predicted_position') is not None
        )

        battles = set()
        for i in range(len(ranked) - 1):
            position, code = ranked[i]
            next_position, next_code = ranked[i + 1]
            # Same threshold the per-driver check used: closer than half a
            # position apart means the two are fighting over the same place.
            if abs(next_position - position) < 0.5:
                battles.add(code)
                battles.add(next_code)

        self._battle_drivers = battles

    def get_trend_indicator(self, driver_code: str) -> tuple:
        """
        Get trend indicator for a driver.

        Args:
            driver_code: Driver abbreviation

        Returns:
            Tuple of (symbol, color) for the trend
        """
        pred = self.predictions.get(driver_code, {})
        trend = pred.get('trend', 'stable')

        if trend == 'improving':
            return ('▲', arcade.color.GREEN)
        elif trend == 'declining':
            return ('▼', arcade.color.RED)
        else:
            return ('►', arcade.color.GRAY)

    def get_position_change(self, driver_code: str) -> Optional[float]:
        """
        Get predicted position change for a driver.

        Args:
            driver_code: Driver abbreviation

        Returns:
            Predicted position change or None
        """
        pred = self.predictions.get(driver_code, {})
        return pred.get('position_change')

    def get_predicted_position(self, driver_code: str) -> Optional[float]:
        """
        Get predicted position for a driver.

        Args:
            driver_code: Driver abbreviation

        Returns:
            Predicted position or None
        """
        pred = self.predictions.get(driver_code, {})
        return pred.get('predicted_position')

    def get_pit_window(self, driver_code: str) -> Optional[tuple]:
        """
        Get pit window for a driver.

        Args:
            driver_code: Driver abbreviation

        Returns:
            Tuple of (start_lap, end_lap) or None
        """
        pred = self.predictions.get(driver_code, {})
        pit_strategy = pred.get('pit_strategy', {})
        return pit_strategy.get('estimated_pit_window')

    def is_in_battle(self, driver_code: str, all_predictions: dict = None) -> bool:
        """
        Check if driver is predicted to be in a battle.

        Answered from the cache built by _rebuild_battle_cache(), so this is a
        set lookup rather than a scan over every other driver.

        Args:
            driver_code: Driver abbreviation
            all_predictions: Unused, kept so existing callers keep working. The
                cache is always derived from self.predictions.

        Returns:
            True if driver is in a predicted battle
        """
        return driver_code in self._battle_drivers

    def draw_leaderboard_overlay(self, x: int, y: int, driver_code: str,
                                  row_height: int = 25):
        """
        Draw prediction overlay next to leaderboard entry.

        Args:
            x: X coordinate for overlay
            y: Y coordinate for leaderboard row
            driver_code: Driver abbreviation
            row_height: Height of the leaderboard row
        """
        if not self.show_overlay:
            return

        # Get trend indicator
        symbol, color = self.get_trend_indicator(driver_code)

        # Draw trend arrow
        arcade.Text(
            symbol,
            x + 5,
            y - row_height / 2,
            color,
            12,
            anchor_x='left',
            anchor_y='center'
        ).draw()

        # Draw predicted position change if available
        pos_change = self.get_position_change(driver_code)
        if pos_change is not None:
            if pos_change < -0.5:
                change_text = f"+{abs(pos_change):.1f}"
                change_color = arcade.color.GREEN
            elif pos_change > 0.5:
                change_text = f"-{abs(pos_change):.1f}"
                change_color = arcade.color.RED
            else:
                change_text = "±0"
                change_color = arcade.color.GRAY

            arcade.Text(
                change_text,
                x + 25,
                y - row_height / 2,
                change_color,
                10,
                anchor_x='left',
                anchor_y='center'
            ).draw()

    def draw_pit_window_indicator(self, x: int, y: int, driver_code: str,
                                   current_lap: int):
        """
        Draw pit window indicator.

        Args:
            x: X coordinate
            y: Y coordinate
            driver_code: Driver abbreviation
            current_lap: Current race lap
        """
        if not self.show_overlay:
            return

        pit_window = self.get_pit_window(driver_code)
        if not pit_window:
            return

        start_lap, end_lap = pit_window

        # Check if in pit window
        if start_lap <= current_lap <= end_lap:
            # Draw pit window indicator
            arcade.draw_circle_filled(x, y, 8, arcade.color.ORANGE)
            arcade.Text(
                'PIT',
                x,
                y,
                arcade.color.BLACK,
                6,
                bold=True,
                anchor_x='center',
                anchor_y='center'
            ).draw()

    def draw_battle_highlight(self, x: int, y: int, driver_code: str):
        """
        Draw battle highlight for driver.

        Args:
            x: X coordinate
            y: Y coordinate
            driver_code: Driver abbreviation
        """
        if not self.show_overlay:
            return

        if self.is_in_battle(driver_code, self.predictions):
            # Draw battle indicator
            arcade.draw_circle_outline(x, y, 12, arcade.color.YELLOW, 2)

    def draw_tables_view(self, screen_width: int, screen_height: int, frame: dict):
        """
        Draw the full tables view when T key is pressed.

        Args:
            screen_width: Screen width
            screen_height: Screen height
            frame: Current race frame
        """
        if not self.show_tables:
            return

        # Draw semi-transparent background
        bg_rect = arcade.XYWH(
            screen_width / 2,
            screen_height / 2,
            screen_width * TABLES_BACKGROUND_WIDTH_RATIO,
            screen_height * TABLES_BACKGROUND_HEIGHT_RATIO
        )
        arcade.draw_rect_filled(bg_rect, (20, 20, 30, 230))
        arcade.draw_rect_outline(bg_rect, arcade.color.CYAN, 2)

        # Draw title
        self._get_text(
            "tables.title",
            "PREDICTION TABLES",
            screen_width / 2,
            screen_height * 0.85,
            arcade.color.CYAN,
            24,
            bold=True,
            anchor_x='center',
            anchor_y='center',
        ).draw()

        # Draw predictions table
        table_x = screen_width * 0.15
        table_y = screen_height * 0.75

        # Headers
        headers = ['Driver', 'Pos', 'Pred', 'Trend', 'Conf']
        col_widths = [80, 50, 50, 80, 60]

        for i, header in enumerate(headers):
            x_pos = table_x + sum(col_widths[:i])
            self._get_text(
                f"tables.header.{i}",
                header,
                x_pos,
                table_y,
                arcade.color.WHITE,
                14,
                bold=True,
                anchor_x='left',
                anchor_y='center',
            ).draw()

        # Draw separator line
        arcade.draw_line(
            table_x, table_y - 15,
            table_x + sum(col_widths), table_y - 15,
            arcade.color.GRAY, 1
        )

        # Draw driver data
        if frame and 'drivers' in frame:
            sorted_drivers = sorted(
                frame['drivers'].items(),
                key=lambda x: x[1].get('position', 99)
            )

            for row_idx, (code, data) in enumerate(sorted_drivers[:15]):
                row_y = table_y - 35 - (row_idx * 25)

                pred = self.predictions.get(code, {})

                values = [
                    code,
                    str(data.get('position', '-')),
                    f"{pred.get('predicted_position', '-'):.1f}" if isinstance(
                        pred.get('predicted_position'), (int, float)) else '-',
                    pred.get('trend', 'stable').upper(),
                    f"{pred.get('confidence', 0) * 100:.0f}%" if pred.get('confidence') else '-'
                ]

                # Trend color
                trend_colors = {
                    'IMPROVING': arcade.color.GREEN,
                    'DECLINING': arcade.color.RED,
                    'STABLE': arcade.color.GRAY
                }

                for col_idx, value in enumerate(values):
                    x_pos = table_x + sum(col_widths[:col_idx])

                    if col_idx == 3:  # Trend column
                        color = trend_colors.get(value, arcade.color.WHITE)
                    else:
                        color = arcade.color.WHITE

                    # Keyed by grid cell: a cell holds its place on screen while
                    # the driver shown in it changes.
                    self._get_text(
                        f"tables.cell.{row_idx}.{col_idx}",
                        value,
                        x_pos,
                        row_y,
                        color,
                        12,
                        anchor_x='left',
                        anchor_y='center',
                    ).draw()

        # Draw instructions
        self._get_text(
            "tables.hint",
            "Press T to close tables view",
            screen_width / 2,
            screen_height * 0.12,
            arcade.color.LIGHT_GRAY,
            14,
            anchor_x='center',
            anchor_y='center',
        ).draw()

    def toggle_overlay(self):
        """Toggle the overlay visibility."""
        self.show_overlay = not self.show_overlay

    def toggle_tables(self):
        """Toggle the tables view."""
        self.show_tables = not self.show_tables
