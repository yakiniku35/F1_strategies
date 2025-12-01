"""
F1 Race Replay with Arcade graphics library.
Based on f1-race-replay project by Tom Shaw.
Extended with ML prediction capabilities.
"""

import os
import arcade
import numpy as np
from typing import Optional
from src.f1_data import FPS
from src.ml_predictor import RaceTrendPredictor
from src.lib.tyres import get_tyre_compound_str
from src.dashboard.prediction_overlay import PredictionOverlay

# Default screen dimensions
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1200
SCREEN_TITLE = "F1 Race Replay with ML Prediction"

# Track status color mapping
STATUS_COLORS = {
    "GREEN": (150, 150, 150),
    "YELLOW": (220, 180, 0),
    "RED": (200, 30, 30),
    "VSC": (200, 130, 50),
    "SC": (180, 100, 30),
    "1": (150, 150, 150),  # Green flag
    "2": (220, 180, 0),    # Yellow flag
    "4": (180, 100, 30),   # Safety car
    "5": (200, 30, 30),    # Red flag
    "6": (200, 130, 50),   # VSC
    "7": (200, 130, 50),   # VSC ending
}


def build_track_from_example_lap(example_lap, track_width=200):
    """Build track geometry from example lap telemetry."""
    plot_x_ref = example_lap["X"].to_numpy()
    plot_y_ref = example_lap["Y"].to_numpy()

    # Compute tangents
    dx = np.gradient(plot_x_ref)
    dy = np.gradient(plot_y_ref)

    norm = np.sqrt(dx**2 + dy**2)
    norm[norm == 0] = 1.0
    dx /= norm
    dy /= norm

    nx = -dy
    ny = dx

    x_outer = plot_x_ref + nx * (track_width / 2)
    y_outer = plot_y_ref + ny * (track_width / 2)
    x_inner = plot_x_ref - nx * (track_width / 2)
    y_inner = plot_y_ref - ny * (track_width / 2)

    # World bounds
    x_min = min(plot_x_ref.min(), x_inner.min(), x_outer.min())
    x_max = max(plot_x_ref.max(), x_inner.max(), x_outer.max())
    y_min = min(plot_y_ref.min(), y_inner.min(), y_outer.min())
    y_max = max(plot_y_ref.max(), y_inner.max(), y_outer.max())

    return (plot_x_ref, plot_y_ref, x_inner, y_inner, x_outer, y_outer,
            x_min, x_max, y_min, y_max)


class F1ReplayWindow(arcade.Window):
    """Main F1 Replay Window with ML prediction integration."""

    def __init__(self, frames, track_statuses, example_lap, drivers, title,
                 playback_speed=1.0, driver_colors=None, predictions: Optional[dict] = None):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, title, resizable=True)

        self.frames = frames
        self.track_statuses = track_statuses
        self.n_frames = len(frames)
        self.drivers = list(drivers)
        self.playback_speed = playback_speed
        self.driver_colors = driver_colors or {}
        self.frame_index = 0.0
        self.paused = False
        self._tyre_textures = {}

        # Load tyre textures
        self._load_tyre_textures()

        # Build track geometry
        (self.plot_x_ref, self.plot_y_ref,
         self.x_inner, self.y_inner,
         self.x_outer, self.y_outer,
         self.x_min, self.x_max,
         self.y_min, self.y_max) = build_track_from_example_lap(example_lap)

        # Pre-calculate interpolated world points
        self.world_inner_points = self._interpolate_points(self.x_inner, self.y_inner)
        self.world_outer_points = self._interpolate_points(self.x_outer, self.y_outer)

        # Screen coordinates
        self.screen_inner_points = []
        self.screen_outer_points = []

        # Scaling parameters
        self.world_scale = 1.0
        self.tx = 0
        self.ty = 0

        # Load background
        bg_path = os.path.join("resources", "background.png")
        self.bg_texture = arcade.load_texture(bg_path) if os.path.exists(bg_path) else None

        arcade.set_background_color(arcade.color.BLACK)

        # Initialize scaling
        self.update_scaling(self.width, self.height)

        # Selection state for leaderboard
        self.selected_driver = None
        self.leaderboard_rects = []

        # ML Prediction
        self.ml_predictor = RaceTrendPredictor()
        self.ml_insights = ["Initializing ML prediction system..."]
        self.ml_trained = False
        self.show_ml_panel = True

        # Prediction overlay
        self.prediction_overlay = PredictionOverlay(predictions)
        self.external_predictions = predictions or {}

        # Train ML model with race data
        self._train_ml_model()

    def _load_tyre_textures(self):
        """Load tyre compound textures."""
        tyres_folder = os.path.join("images", "tyres")
        if os.path.exists(tyres_folder):
            for filename in os.listdir(tyres_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    texture_name = os.path.splitext(filename)[0]
                    texture_path = os.path.join(tyres_folder, filename)
                    try:
                        self._tyre_textures[texture_name] = arcade.load_texture(texture_path)
                    except Exception:
                        pass

    def _train_ml_model(self):
        """Train the ML prediction model with race data."""
        print("Training ML prediction model...")
        if self.ml_predictor.train(self.frames, self.drivers):
            self.ml_trained = True
            self.ml_insights = ["ML model trained successfully!"]
            print("ML model trained successfully!")
        else:
            self.ml_insights = ["ML training failed - insufficient data"]
            print("ML training failed")

    def _interpolate_points(self, xs, ys, interp_points=2000):
        """Generate smooth points in world coordinates."""
        t_old = np.linspace(0, 1, len(xs))
        t_new = np.linspace(0, 1, interp_points)
        xs_i = np.interp(t_new, t_old, xs)
        ys_i = np.interp(t_new, t_old, ys)
        return list(zip(xs_i, ys_i))

    def update_scaling(self, screen_w, screen_h):
        """Recalculate scale and translation for the track."""
        padding = 0.05
        world_w = max(1.0, self.x_max - self.x_min)
        world_h = max(1.0, self.y_max - self.y_min)

        usable_w = screen_w * (1 - 2 * padding)
        usable_h = screen_h * (1 - 2 * padding)

        scale_x = usable_w / world_w
        scale_y = usable_h / world_h
        self.world_scale = min(scale_x, scale_y)

        world_cx = (self.x_min + self.x_max) / 2
        world_cy = (self.y_min + self.y_max) / 2
        screen_cx = screen_w / 2
        screen_cy = screen_h / 2

        self.tx = screen_cx - self.world_scale * world_cx
        self.ty = screen_cy - self.world_scale * world_cy

        # Update polyline screen coordinates
        self.screen_inner_points = [self.world_to_screen(x, y) for x, y in self.world_inner_points]
        self.screen_outer_points = [self.world_to_screen(x, y) for x, y in self.world_outer_points]

    def on_resize(self, width, height):
        """Handle window resize."""
        super().on_resize(width, height)
        self.update_scaling(width, height)

    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates."""
        sx = self.world_scale * x + self.tx
        sy = self.world_scale * y + self.ty
        return sx, sy

    def on_draw(self):
        """Render the frame."""
        self.clear()

        # 1. Draw Background
        if self.bg_texture:
            arcade.draw_lrbt_rectangle_textured(
                left=0, right=self.width,
                bottom=0, top=self.height,
                texture=self.bg_texture
            )

        # 2. Get current frame data
        idx = min(int(self.frame_index), self.n_frames - 1)
        frame = self.frames[idx]
        current_time = frame["t"]

        # Get current track status
        current_track_status = "1"  # Default green
        for status in self.track_statuses:
            if status['start_time'] <= current_time:
                if status['end_time'] is None or current_time < status['end_time']:
                    current_track_status = status['status']
                    break

        track_color = STATUS_COLORS.get(current_track_status, (150, 150, 150))

        # 3. Draw Track
        if len(self.screen_inner_points) > 1:
            arcade.draw_line_strip(self.screen_inner_points, track_color, 4)
        if len(self.screen_outer_points) > 1:
            arcade.draw_line_strip(self.screen_outer_points, track_color, 4)

        # 4. Draw Cars
        for code, pos in frame["drivers"].items():
            if pos.get("rel_dist", 0) == 1:
                continue
            sx, sy = self.world_to_screen(pos["x"], pos["y"])
            color = self.driver_colors.get(code, arcade.color.WHITE)
            arcade.draw_circle_filled(sx, sy, 6, color)

        # --- UI ELEMENTS ---
        self._draw_hud(frame, current_time, current_track_status)
        self._draw_leaderboard(frame)
        self._draw_prediction_overlay(frame)
        self._draw_controls_legend()
        self._draw_selected_driver_info(frame)
        self._draw_ml_panel(frame)

        # Draw tables view (on top of everything when active)
        self.prediction_overlay.draw_tables_view(self.width, self.height, frame)

    def _draw_hud(self, frame, current_time, track_status):
        """Draw heads-up display (lap, time, flags)."""
        # Get leader info
        leader_code = max(
            frame["drivers"],
            key=lambda c: (frame["drivers"][c].get("lap", 1), frame["drivers"][c].get("dist", 0))
        )
        leader_lap = frame["drivers"][leader_code].get("lap", 1)

        # Time calculation
        t = current_time
        hours = int(t // 3600)
        minutes = int((t % 3600) // 60)
        seconds = int(t % 60)
        time_str = f"{hours:02}:{minutes:02}:{seconds:02}"

        # Draw HUD - Top Left
        arcade.Text(f"Lap: {leader_lap}",
                    20, self.height - 40,
                    arcade.color.WHITE, 24, anchor_y="top").draw()

        arcade.Text(f"Race Time: {time_str}",
                    20, self.height - 80,
                    arcade.color.WHITE, 20, anchor_y="top").draw()

        arcade.Text(f"Speed: {self.playback_speed}x",
                    20, self.height - 120,
                    arcade.color.LIGHT_GRAY, 16, anchor_y="top").draw()

        # Track status flag
        status_texts = {
            "2": ("YELLOW FLAG", arcade.color.YELLOW),
            "4": ("SAFETY CAR", arcade.color.BROWN),
            "5": ("RED FLAG", arcade.color.RED),
            "6": ("VIRTUAL SAFETY CAR", arcade.color.ORANGE),
            "7": ("VSC ENDING", arcade.color.ORANGE),
        }

        if track_status in status_texts:
            text, color = status_texts[track_status]
            arcade.Text(text,
                        20, self.height - 160,
                        color, 24, bold=True, anchor_y="top").draw()

    def _draw_leaderboard(self, frame):
        """Draw the leaderboard on the right side."""
        leaderboard_x = self.width - 220
        leaderboard_y = self.height - 40

        arcade.Text("Leaderboard", leaderboard_x, leaderboard_y,
                    arcade.color.WHITE, 20, bold=True, anchor_x="left", anchor_y="top").draw()

        driver_list = []
        for code, pos in frame["drivers"].items():
            color = self.driver_colors.get(code, arcade.color.WHITE)
            driver_list.append((code, color, pos))

        # Sort by distance
        driver_list.sort(key=lambda x: x[2].get("dist", 999), reverse=True)

        self.leaderboard_rects = []
        row_height = 25
        entry_width = 240

        for i, (code, color, pos) in enumerate(driver_list):
            current_pos = i + 1
            if pos.get("rel_dist", 0) == 1:
                text = f"{current_pos}. {code}   OUT"
            else:
                text = f"{current_pos}. {code}"

            top_y = leaderboard_y - 30 - (i * row_height)
            bottom_y = top_y - row_height
            left_x = leaderboard_x
            right_x = leaderboard_x + entry_width

            self.leaderboard_rects.append((code, left_x, bottom_y, right_x, top_y))

            # Highlight if selected
            if code == self.selected_driver:
                rect = arcade.XYWH(
                    (left_x + right_x) / 2,
                    (top_y + bottom_y) / 2,
                    right_x - left_x,
                    top_y - bottom_y
                )
                arcade.draw_rect_filled(rect, arcade.color.LIGHT_GRAY)
                text_color = arcade.color.BLACK
            else:
                text_color = color

            arcade.Text(
                text, left_x, top_y,
                text_color, 16,
                anchor_x="left", anchor_y="top"
            ).draw()

            # Tyre icon
            tyre_name = get_tyre_compound_str(pos.get("tyre", 1))
            tyre_texture = self._tyre_textures.get(tyre_name.upper())
            if tyre_texture:
                tyre_icon_x = self.width - 30
                tyre_icon_y = top_y - 12
                icon_size = 16
                rect = arcade.XYWH(tyre_icon_x, tyre_icon_y, icon_size, icon_size)
                arcade.draw_texture_rect(rect=rect, texture=tyre_texture, angle=0, alpha=255)

    def _draw_prediction_overlay(self, frame):
        """Draw prediction overlay elements on the leaderboard."""
        if not self.prediction_overlay.show_overlay:
            return

        # Update predictions from ML model if available
        if self.ml_trained and int(self.frame_index) % 25 == 0:
            live_predictions = self.ml_predictor.predict_all_drivers(frame)
            if live_predictions:
                # Merge with external predictions
                merged = {**self.external_predictions, **live_predictions}
                self.prediction_overlay.update_predictions(merged)

        # Draw overlay for each leaderboard entry
        leaderboard_x = self.width - 220
        leaderboard_y = self.height - 40
        row_height = 25
        overlay_x = self.width - 55

        driver_list = []
        for code, pos in frame["drivers"].items():
            driver_list.append((code, pos))
        driver_list.sort(key=lambda x: x[1].get("dist", 999), reverse=True)

        for i, (code, pos) in enumerate(driver_list):
            top_y = leaderboard_y - 30 - (i * row_height)

            # Draw trend indicator
            self.prediction_overlay.draw_leaderboard_overlay(
                overlay_x, top_y, code, row_height
            )

            # Draw pit window indicator
            current_lap = pos.get('lap', 1)
            self.prediction_overlay.draw_pit_window_indicator(
                self.width - 75, top_y - 12, code, current_lap
            )

            # Draw battle highlight on car position
            if code in frame["drivers"]:
                sx, sy = self.world_to_screen(pos["x"], pos["y"])
                self.prediction_overlay.draw_battle_highlight(sx, sy, code)

    def _draw_controls_legend(self):
        """Draw controls legend at bottom left."""
        legend_x = 20
        legend_y = 175
        legend_lines = [
            "Controls:",
            "[SPACE]  Pause/Resume",
            "[←/→]    Rewind / FastForward",
            "[↑/↓]    Speed +/- (0.5x, 1x, 2x, 4x)",
            "[M]      Toggle ML Panel",
            "[T]      Toggle Tables View",
        ]

        for i, line in enumerate(legend_lines):
            arcade.Text(
                line,
                legend_x,
                legend_y - (i * 25),
                arcade.color.LIGHT_GRAY if i > 0 else arcade.color.WHITE,
                14,
                bold=(i == 0)
            ).draw()

    def _draw_selected_driver_info(self, frame):
        """Draw selected driver information panel."""
        if not self.selected_driver or self.selected_driver not in frame["drivers"]:
            return

        driver_pos = frame["drivers"][self.selected_driver]
        driver_color = self.driver_colors.get(self.selected_driver, arcade.color.GRAY)

        info_x = 20
        info_y = self.height / 2 + 100
        box_width = 300
        box_height = 180

        # Background box
        bg_rect = arcade.XYWH(
            info_x + box_width / 2,
            info_y - box_height / 2,
            box_width,
            box_height
        )
        arcade.draw_rect_outline(bg_rect, driver_color)

        # Driver name box
        name_rect = arcade.XYWH(
            info_x + box_width / 2,
            info_y + 20,
            box_width,
            40
        )
        arcade.draw_rect_filled(name_rect, driver_color)
        arcade.Text(
            f"Driver: {self.selected_driver}",
            info_x + 10,
            info_y + 20,
            arcade.color.BLACK,
            16,
            anchor_x="left", anchor_y="center"
        ).draw()

        # Driver stats
        speed_text = f"Speed: {driver_pos.get('speed', 0):.1f} km/h"
        gear_text = f"Gear: {driver_pos.get('gear', 0)}"

        drs_value = driver_pos.get('drs', 0)
        if drs_value in [0, 1]:
            drs_status = "Off"
        elif drs_value == 8:
            drs_status = "Eligible"
        elif drs_value in [10, 12, 14]:
            drs_status = "On"
        else:
            drs_status = "Unknown"
        drs_text = f"DRS: {drs_status}"

        lap_text = f"Current Lap: {driver_pos.get('lap', 1)}"
        tyre_text = f"Tyre: {get_tyre_compound_str(driver_pos.get('tyre', 1))}"

        # ML Prediction for selected driver
        prediction = self.ml_predictor.predict(frame, self.selected_driver) if self.ml_trained else None
        if prediction:
            pred_text = f"Predicted: P{prediction['predicted_position']:.0f} ({prediction['trend']})"
        else:
            pred_text = "Prediction: N/A"

        stats_lines = [speed_text, gear_text, drs_text, lap_text, tyre_text, pred_text]
        for i, line in enumerate(stats_lines):
            arcade.Text(
                line,
                info_x + 10,
                info_y - 20 - (i * 25),
                arcade.color.WHITE,
                14,
                anchor_x="left", anchor_y="center"
            ).draw()

    def _draw_ml_panel(self, frame):
        """Draw ML prediction panel."""
        if not self.show_ml_panel:
            return

        panel_x = self.width - 450
        panel_y = 200
        panel_width = 420
        panel_height = 180

        # Panel background
        bg_rect = arcade.XYWH(
            panel_x + panel_width / 2,
            panel_y + panel_height / 2,
            panel_width,
            panel_height
        )
        arcade.draw_rect_filled(bg_rect, (30, 30, 40, 200))
        arcade.draw_rect_outline(bg_rect, arcade.color.CYAN)

        # Panel title
        arcade.Text(
            "🤖 ML Race Prediction",
            panel_x + 10,
            panel_y + panel_height - 20,
            arcade.color.CYAN,
            18,
            bold=True,
            anchor_x="left", anchor_y="center"
        ).draw()

        # Update insights periodically
        if int(self.frame_index) % 50 == 0 and self.ml_trained:
            self.ml_insights = self.ml_predictor.get_race_insights(frame)

        # Draw insights
        for i, insight in enumerate(self.ml_insights[:4]):
            arcade.Text(
                insight,
                panel_x + 15,
                panel_y + panel_height - 55 - (i * 30),
                arcade.color.WHITE,
                13,
                anchor_x="left", anchor_y="center"
            ).draw()

    def on_update(self, delta_time: float):
        """Update game state."""
        if self.paused:
            return
        self.frame_index += delta_time * FPS * self.playback_speed
        if self.frame_index >= self.n_frames:
            self.frame_index = float(self.n_frames - 1)

    def on_key_press(self, symbol: int, modifiers: int):
        """Handle keyboard input."""
        if symbol == arcade.key.SPACE:
            self.paused = not self.paused
        elif symbol == arcade.key.RIGHT:
            self.frame_index = min(self.frame_index + 10.0, self.n_frames - 1)
        elif symbol == arcade.key.LEFT:
            self.frame_index = max(self.frame_index - 10.0, 0.0)
        elif symbol == arcade.key.UP:
            self.playback_speed = min(self.playback_speed * 2.0, 8.0)
        elif symbol == arcade.key.DOWN:
            self.playback_speed = max(0.25, self.playback_speed / 2.0)
        elif symbol == arcade.key.KEY_1:
            self.playback_speed = 0.5
        elif symbol == arcade.key.KEY_2:
            self.playback_speed = 1.0
        elif symbol == arcade.key.KEY_3:
            self.playback_speed = 2.0
        elif symbol == arcade.key.KEY_4:
            self.playback_speed = 4.0
        elif symbol == arcade.key.M:
            self.show_ml_panel = not self.show_ml_panel
        elif symbol == arcade.key.T:
            self.prediction_overlay.toggle_tables()

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int):
        """Handle mouse click for driver selection."""
        new_selection = None
        for code, left, bottom, right, top in self.leaderboard_rects:
            if left <= x <= right and bottom <= y <= top:
                new_selection = code
                break

        if new_selection == self.selected_driver:
            self.selected_driver = None
        else:
            self.selected_driver = new_selection


def run_arcade_replay(frames, track_statuses, example_lap, drivers, title,
                      playback_speed=1.0, driver_colors=None, predictions=None):
    """Run the F1 replay visualization.

    Args:
        frames: Race telemetry frames
        track_statuses: Track status data
        example_lap: Example lap for track geometry
        drivers: List of driver codes
        title: Window title
        playback_speed: Initial playback speed multiplier
        driver_colors: Dictionary mapping driver codes to RGB colors
        predictions: Optional dictionary of ML predictions
    """
    F1ReplayWindow(
        frames=frames,
        track_statuses=track_statuses,
        example_lap=example_lap,
        drivers=drivers,
        playback_speed=playback_speed,
        driver_colors=driver_colors,
        title=title,
        predictions=predictions
    )
    arcade.run()
