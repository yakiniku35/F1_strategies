"""
F1 Race Replay with Arcade graphics library.
Based on f1-race-replay project by Tom Shaw.
Extended with ML prediction capabilities and AI chat.

Optimized for performance with batch rendering and NumPy data access.
"""

import os
import arcade
import numpy as np
from typing import Optional, Union, List
from src.f1_data import (
    FPS, FIELD_X, FIELD_Y, FIELD_DIST, FIELD_REL_DIST, FIELD_LAP,
    FIELD_TYRE, FIELD_SPEED, FIELD_GEAR, FIELD_DRS, FIELD_POSITION
)
from src.ml_predictor import RaceTrendPredictor
from src.lib.tyres import get_tyre_compound_str
from src.dashboard.prediction_overlay import PredictionOverlay
from src.ai_chat import F1AIChat

# Default screen dimensions (1080p for better compatibility)
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
SCREEN_TITLE = "F1 Race Replay with ML Prediction"

# UI Layout Constants
LEADERBOARD_WIDTH = 280
LEADERBOARD_PADDING = 15
HUD_PANEL_WIDTH = 250
HUD_PANEL_HEIGHT = 180
ML_PANEL_WIDTH = 440
ML_PANEL_HEIGHT = 200

# Chat UI Constants
CHAT_PANEL_WIDTH = 500
CHAT_PANEL_HEIGHT = 400
CHAT_INPUT_HEIGHT = 40

# HUD Spacing Constants
HUD_LINE_HEIGHT = 30
HUD_SECTION_GAP = 45
ML_INSIGHT_MAX_LENGTH = 55

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

# Panel colors
PANEL_BG_COLOR = (20, 20, 30, 200)
PANEL_BORDER_COLOR = (60, 60, 80)
HEADER_BG_COLOR = (40, 40, 60, 220)


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
    """Main F1 Replay Window with ML prediction integration.
    
    Supports both legacy frame format (list of dicts) and optimized NumPy arrays.
    When using NumPy arrays, provides significant performance improvements through
    batch rendering and vectorized data access.
    """

    def __init__(self, frames, track_statuses, example_lap, drivers, title,
                 playback_speed=1.0, driver_colors=None, predictions: Optional[dict] = None,
                 mode: str = 'historical', race_info: Optional[dict] = None,
                 driver_data_array: Optional[np.ndarray] = None,
                 frame_metadata: Optional[np.ndarray] = None,
                 driver_codes: Optional[List[str]] = None):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, title, resizable=True)

        self.track_statuses = track_statuses
        self.playback_speed = playback_speed
        self.driver_colors = driver_colors or {}
        self.frame_index = 0.0
        self.paused = False
        self._tyre_textures = {}

        # Mode: 'historical' or 'predicted'
        self.mode = mode
        self.race_info = race_info or {}

        # Optimized NumPy array format for performance
        self.use_numpy_arrays = driver_data_array is not None
        if self.use_numpy_arrays:
            self.driver_data_array = driver_data_array
            self.frame_metadata = frame_metadata
            self.driver_codes = driver_codes or list(drivers)
            self.n_frames = driver_data_array.shape[0]
            self.n_drivers = driver_data_array.shape[1]
            self.drivers = self.driver_codes
            # Create driver code to index mapping for fast lookup
            self._driver_idx_map = {code: idx for idx, code in enumerate(self.driver_codes)}
            # Legacy frames not needed
            self.frames = None
        else:
            # Legacy frame format
            self.frames = frames
            self.n_frames = len(frames)
            self.drivers = list(drivers)
            self.driver_data_array = None
            self.frame_metadata = None
            self.driver_codes = None
            self._driver_idx_map = None

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

        # Batch rendering - Static track shapes (regenerated only when track status changes)
        self._track_shapes: Optional[arcade.ShapeElementList] = None
        self._last_track_status = None  # Track the last status to detect changes

        # Batch rendering - Car sprites
        self._car_sprites: Optional[arcade.SpriteList] = None
        self._car_sprite_map: dict = {}  # Maps driver code to sprite
        self._init_car_sprites()

        # ML Prediction
        self.ml_predictor = RaceTrendPredictor()
        self.ml_insights = ["Initializing ML prediction system..."]
        self.ml_trained = False
        self.show_ml_panel = True

        # Prediction overlay
        self.prediction_overlay = PredictionOverlay(predictions)
        self.external_predictions = predictions or {}

        # AI Chat
        self.ai_chat = F1AIChat()
        self.ai_chat.set_race_context(race_info or {}, self.drivers)
        self.show_chat_panel = False
        self.chat_input = ""
        self.chat_messages = []
        self.chat_input_active = False

        # For predicted mode, skip training on frame data
        if self.mode != 'predicted':
            self._train_ml_model()
        else:
            self.ml_insights = ["🔮 Running in prediction mode"]
            self.ml_trained = True
            # Store last update time for predicted insights
            self._last_insight_update = 0

    def _init_car_sprites(self):
        """Initialize car sprites for batch rendering."""
        self._car_sprites = arcade.SpriteList()
        self._car_sprite_map = {}
        
        for code in self.drivers:
            color = self.driver_colors.get(code, arcade.color.WHITE)
            # Create a simple circular sprite for each car
            sprite = arcade.SpriteCircle(8, color)
            sprite.visible = False  # Start hidden, will be positioned in on_update
            self._car_sprites.append(sprite)
            self._car_sprite_map[code] = sprite

    def _build_track_shapes(self, track_color):
        """Build static track geometry as ShapeElementList for batch rendering.
        
        Only regenerate when track status (color) changes.
        """
        self._track_shapes = arcade.ShapeElementList()
        
        # Create line strips for inner and outer track edges
        if len(self.screen_inner_points) > 1:
            inner_line = arcade.create_line_strip(
                self.screen_inner_points, track_color, 3
            )
            self._track_shapes.append(inner_line)
        
        if len(self.screen_outer_points) > 1:
            outer_line = arcade.create_line_strip(
                self.screen_outer_points, track_color, 3
            )
            self._track_shapes.append(outer_line)
        
        self._last_track_status = track_color

    def _get_current_frame_data(self, frame_idx: int) -> dict:
        """Get current frame data, supporting both legacy and NumPy formats.
        
        Returns a frame dictionary compatible with legacy code.
        """
        if self.use_numpy_arrays:
            # Fast NumPy array access
            frame_data = {}
            for driver_idx, code in enumerate(self.driver_codes):
                frame_data[code] = {
                    "x": float(self.driver_data_array[frame_idx, driver_idx, FIELD_X]),
                    "y": float(self.driver_data_array[frame_idx, driver_idx, FIELD_Y]),
                    "dist": float(self.driver_data_array[frame_idx, driver_idx, FIELD_DIST]),
                    "rel_dist": float(self.driver_data_array[frame_idx, driver_idx, FIELD_REL_DIST]),
                    "lap": int(round(self.driver_data_array[frame_idx, driver_idx, FIELD_LAP])),
                    "tyre": int(self.driver_data_array[frame_idx, driver_idx, FIELD_TYRE]),
                    "position": int(self.driver_data_array[frame_idx, driver_idx, FIELD_POSITION]),
                    "speed": float(self.driver_data_array[frame_idx, driver_idx, FIELD_SPEED]),
                    "gear": int(self.driver_data_array[frame_idx, driver_idx, FIELD_GEAR]),
                    "drs": int(self.driver_data_array[frame_idx, driver_idx, FIELD_DRS]),
                }
            return {
                "t": float(self.frame_metadata[frame_idx, 0]),
                "lap": int(self.frame_metadata[frame_idx, 1]),
                "drivers": frame_data,
            }
        else:
            # Legacy frame format
            return self.frames[frame_idx]

    def _update_car_sprites(self, frame: dict):
        """Update car sprite positions for batch rendering."""
        for code, pos in frame["drivers"].items():
            if code not in self._car_sprite_map:
                continue
            
            sprite = self._car_sprite_map[code]
            
            # Hide cars that are out
            if pos.get("rel_dist", 0) == 1:
                sprite.visible = False
                continue
            
            sprite.visible = True
            sx, sy = self.world_to_screen(pos["x"], pos["y"])
            sprite.center_x = sx
            sprite.center_y = sy

    def _generate_predicted_insights(self, frame):
        """Generate insights for predicted mode based on current frame data."""
        insights = []
        lap = frame.get('lap', 1)
        
        # Get race info
        gp_name = self.race_info.get('gp', 'Race')
        year = self.race_info.get('year', 2025)
        
        # Sort drivers by position
        sorted_drivers = sorted(
            frame['drivers'].items(),
            key=lambda x: x[1].get('position', 99)
        )
        
        if len(sorted_drivers) >= 3:
            leader = sorted_drivers[0]
            second = sorted_drivers[1]
            third = sorted_drivers[2]
            
            # Leader insight
            insights.append(f"🏆 {leader[0]} leads the {year} {gp_name}")
            
            # Battle for positions
            if len(sorted_drivers) >= 2:
                leader_dist = leader[1].get('dist', 0)
                second_dist = second[1].get('dist', 0)
                gap = leader_dist - second_dist
                if gap < 200:
                    insights.append(f"⚔️ Close battle: {leader[0]} vs {second[0]} for P1!")
                elif gap < 500:
                    insights.append(f"🔥 {second[0]} closing in on {leader[0]}")
            
            # Podium positions
            insights.append(f"🥇🥈🥉 Podium: {leader[0]}, {second[0]}, {third[0]}")
            
            # Lap progress
            total_laps = self.race_info.get('total_laps', 50)
            if lap < total_laps * 0.25:
                insights.append(f"📊 Lap {lap} - Early race phase")
            elif lap < total_laps * 0.75:
                insights.append(f"📊 Lap {lap} - Mid-race, strategy in play")
            else:
                insights.append(f"📊 Lap {lap} - Final laps, push to finish!")
        
        return insights[:4] if insights else ["🔮 Simulating predicted race..."]

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
        
        # If using NumPy arrays, convert to frames for ML training
        if self.use_numpy_arrays:
            # ML predictor can accept NumPy arrays directly
            if self.ml_predictor.train_from_numpy(
                self.driver_data_array, self.frame_metadata, self.driver_codes
            ):
                self.ml_trained = True
                self.ml_insights = ["ML model trained successfully!"]
                print("ML model trained successfully!")
            else:
                self.ml_insights = ["ML training failed - insufficient data"]
                print("ML training failed")
        else:
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
        # Invalidate track shapes cache so they get rebuilt with new coordinates
        self._track_shapes = None
        self._last_track_status = None

    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates."""
        sx = self.world_scale * x + self.tx
        sy = self.world_scale * y + self.ty
        return sx, sy

    def on_draw(self):
        """Render the frame with optimized batch rendering."""
        self.clear()

        # 1. Draw Background
        if self.bg_texture:
            arcade.draw_texture_rect(
                texture=self.bg_texture,
                rect=arcade.LBWH(0, 0, self.width, self.height)
            )

        # 2. Get current frame data (supports both legacy and NumPy formats)
        idx = min(int(self.frame_index), self.n_frames - 1)
        frame = self._get_current_frame_data(idx)
        current_time = frame["t"]

        # Get current track status
        current_track_status = "1"  # Default green
        for status in self.track_statuses:
            if status['start_time'] <= current_time:
                if status['end_time'] is None or current_time < status['end_time']:
                    current_track_status = status['status']
                    break

        track_color = STATUS_COLORS.get(current_track_status, (150, 150, 150))

        # 3. Draw Track using batch rendering (ShapeElementList)
        # Only rebuild if track status/color changed or shapes don't exist
        if self._track_shapes is None or self._last_track_status != track_color:
            self._build_track_shapes(track_color)
        
        if self._track_shapes:
            self._track_shapes.draw()

        # 4. Draw Cars - use batch SpriteList for better performance
        # Update sprite positions
        self._update_car_sprites(frame)
        
        # Draw all car sprites in one batch call
        self._car_sprites.draw()
        
        # Draw car outlines and labels (these still need individual calls)
        for code, pos in frame["drivers"].items():
            if pos.get("rel_dist", 0) == 1:
                continue
            sx, sy = self.world_to_screen(pos["x"], pos["y"])

            # Draw car outline
            arcade.draw_circle_outline(sx, sy, 8, arcade.color.WHITE, 2)

            # Draw driver code label for selected driver or top 3
            position = pos.get("position", 99)
            if code == self.selected_driver or position <= 3:
                # Background for label
                arcade.draw_rect_filled(arcade.XYWH(sx, sy + 18, 28, 14), (0, 0, 0, 180))
                arcade.Text(
                    code,
                    sx, sy + 18,
                    arcade.color.WHITE,
                    9,
                    bold=True,
                    anchor_x="center", anchor_y="center"
                ).draw()

        # --- UI ELEMENTS ---
        self._draw_hud(frame, current_time, current_track_status)
        self._draw_leaderboard(frame)
        self._draw_prediction_overlay(frame)
        self._draw_controls_legend()
        self._draw_selected_driver_info(frame)
        self._draw_ml_panel(frame)

        # Draw tables view (on top of everything when active)
        self.prediction_overlay.draw_tables_view(self.width, self.height, frame)

        # Draw chat panel (on top of everything when active)
        self._draw_chat_panel(frame)

    def _draw_hud(self, frame, current_time, track_status):
        """Draw heads-up display (lap, time, flags) with panel background."""
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

        # Draw HUD panel background
        panel_x = LEADERBOARD_PADDING
        panel_y = self.height - LEADERBOARD_PADDING - HUD_PANEL_HEIGHT
        bg_rect = arcade.XYWH(
            panel_x + HUD_PANEL_WIDTH / 2,
            panel_y + HUD_PANEL_HEIGHT / 2,
            HUD_PANEL_WIDTH,
            HUD_PANEL_HEIGHT
        )
        arcade.draw_rect_filled(bg_rect, PANEL_BG_COLOR)
        arcade.draw_rect_outline(bg_rect, PANEL_BORDER_COLOR, 2)

        # Draw predicted race indicator if in predicted mode
        if self.mode == 'predicted':
            gp_name = self.race_info.get('gp', 'Unknown GP')
            year = self.race_info.get('year', 2025)
            banner_text = f"🔮 PREDICTED - {year} {gp_name}"
            arcade.Text(banner_text,
                        self.width / 2, self.height - 15,
                        arcade.color.CYAN, 18, bold=True,
                        anchor_x="center", anchor_y="top").draw()

        # HUD content
        text_x = panel_x + 15
        text_y = self.height - LEADERBOARD_PADDING - 20
        line_offset = 0

        # Lap counter with larger font
        arcade.Text(f"LAP {leader_lap}",
                    text_x, text_y - line_offset,
                    arcade.color.WHITE, 28, bold=True, anchor_y="top").draw()
        line_offset += HUD_SECTION_GAP

        # Race time
        arcade.Text(f"⏱ {time_str}",
                    text_x, text_y - line_offset,
                    arcade.color.LIGHT_GRAY, 18, anchor_y="top").draw()
        line_offset += HUD_LINE_HEIGHT

        # Playback speed
        speed_color = arcade.color.GREEN if self.playback_speed > 1 else (
            arcade.color.YELLOW if self.playback_speed < 1 else arcade.color.WHITE
        )
        arcade.Text(f"▶ {self.playback_speed}x",
                    text_x, text_y - line_offset,
                    speed_color, 16, anchor_y="top").draw()
        line_offset += HUD_LINE_HEIGHT

        # Pause indicator
        if self.paused:
            arcade.Text("⏸ PAUSED",
                        text_x, text_y - line_offset,
                        arcade.color.YELLOW, 16, bold=True, anchor_y="top").draw()
        line_offset += HUD_LINE_HEIGHT

        # Track status flag with background
        status_texts = {
            "2": ("⚠ YELLOW FLAG", arcade.color.YELLOW, (80, 80, 0)),
            "4": ("🚗 SAFETY CAR", arcade.color.ORANGE, (80, 40, 0)),
            "5": ("🛑 RED FLAG", arcade.color.RED, (80, 0, 0)),
            "6": ("⚡ VSC", arcade.color.ORANGE, (80, 50, 0)),
            "7": ("⚡ VSC ENDING", arcade.color.ORANGE, (80, 50, 0)),
        }

        if track_status in status_texts:
            text, color, bg_color = status_texts[track_status]
            flag_y = text_y - line_offset
            flag_rect = arcade.XYWH(
                text_x + 100, flag_y + 10,
                200, 30
            )
            arcade.draw_rect_filled(flag_rect, bg_color)
            arcade.Text(text,
                        text_x + 5, flag_y,
                        color, 18, bold=True, anchor_y="top").draw()

    def _draw_leaderboard(self, frame):
        """Draw the leaderboard on the right side with panel background."""
        num_drivers = len(frame["drivers"])
        row_height = 26
        header_height = 40
        leaderboard_height = header_height + (num_drivers * row_height) + 20

        # Position leaderboard
        leaderboard_x = self.width - LEADERBOARD_WIDTH - LEADERBOARD_PADDING
        leaderboard_y = self.height - LEADERBOARD_PADDING

        # Draw panel background
        bg_rect = arcade.XYWH(
            leaderboard_x + LEADERBOARD_WIDTH / 2,
            leaderboard_y - leaderboard_height / 2,
            LEADERBOARD_WIDTH,
            leaderboard_height
        )
        arcade.draw_rect_filled(bg_rect, PANEL_BG_COLOR)
        arcade.draw_rect_outline(bg_rect, PANEL_BORDER_COLOR, 2)

        # Draw header background
        header_rect = arcade.XYWH(
            leaderboard_x + LEADERBOARD_WIDTH / 2,
            leaderboard_y - header_height / 2,
            LEADERBOARD_WIDTH,
            header_height
        )
        arcade.draw_rect_filled(header_rect, HEADER_BG_COLOR)

        # Header text
        arcade.Text("🏁 LIVE STANDINGS",
                    leaderboard_x + 10, leaderboard_y - 12,
                    arcade.color.WHITE, 16, bold=True,
                    anchor_x="left", anchor_y="top").draw()

        # Prepare driver list
        driver_list = []
        for code, pos in frame["drivers"].items():
            color = self.driver_colors.get(code, arcade.color.WHITE)
            driver_list.append((code, color, pos))

        # Sort by distance (race position)
        driver_list.sort(key=lambda x: x[2].get("dist", 999), reverse=True)

        # Get leader distance for gap calculation
        if driver_list:
            leader_dist = driver_list[0][2].get("dist", 0)

        self.leaderboard_rects = []

        for i, (code, color, pos) in enumerate(driver_list):
            current_pos = i + 1
            top_y = leaderboard_y - header_height - 5 - (i * row_height)
            bottom_y = top_y - row_height
            left_x = leaderboard_x + 5
            right_x = leaderboard_x + LEADERBOARD_WIDTH - 5

            self.leaderboard_rects.append((code, left_x, bottom_y, right_x, top_y))

            # Highlight if selected
            if code == self.selected_driver:
                highlight_rect = arcade.XYWH(
                    leaderboard_x + LEADERBOARD_WIDTH / 2,
                    (top_y + bottom_y) / 2,
                    LEADERBOARD_WIDTH - 10,
                    row_height - 2
                )
                arcade.draw_rect_filled(highlight_rect, (70, 70, 100, 180))

            # Position number with background
            pos_color = (255, 215, 0) if current_pos <= 3 else (100, 100, 100)
            arcade.draw_circle_filled(left_x + 12, top_y - row_height / 2 + 2, 10, pos_color)
            arcade.Text(str(current_pos),
                        left_x + 12, top_y - row_height / 2 + 2,
                        arcade.color.BLACK if current_pos <= 3 else arcade.color.WHITE,
                        11, bold=True, anchor_x="center", anchor_y="center").draw()

            # Driver code with team color indicator
            arcade.draw_rect_filled(arcade.XYWH(left_x + 30, top_y - row_height / 2 + 2, 4, 16), color)

            # Check if OUT
            is_out = pos.get("rel_dist", 0) == 1
            text_color = arcade.color.GRAY if is_out else arcade.color.WHITE

            driver_text = f"{code}" + (" OUT" if is_out else "")
            arcade.Text(driver_text,
                        left_x + 40, top_y - row_height / 2 + 2,
                        text_color, 13, bold=True,
                        anchor_x="left", anchor_y="center").draw()

            # Gap to leader
            if i > 0 and not is_out:
                gap = leader_dist - pos.get("dist", 0)
                gap_text = f"+{gap / 1000:.1f}km" if gap > 1000 else f"+{gap:.0f}m"
                arcade.Text(gap_text,
                            right_x - 60, top_y - row_height / 2 + 2,
                            arcade.color.LIGHT_GRAY, 10,
                            anchor_x="right", anchor_y="center").draw()

            # Tyre icon
            tyre_name = get_tyre_compound_str(pos.get("tyre", 1))
            tyre_texture = self._tyre_textures.get(tyre_name.upper())
            if tyre_texture:
                tyre_icon_x = right_x - 18
                tyre_icon_y = top_y - row_height / 2 + 2
                icon_size = 18
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

        # Draw battle highlights on car positions
        for code, pos in frame["drivers"].items():
            sx, sy = self.world_to_screen(pos["x"], pos["y"])
            self.prediction_overlay.draw_battle_highlight(sx, sy, code)

    def _draw_controls_legend(self):
        """Draw controls legend at bottom left with panel background."""
        legend_lines = [
            "⌨ CONTROLS",
            "SPACE  Pause/Resume",
            "← / →  Rewind / Forward",
            "↑ / ↓  Speed +/-",
            "M      Toggle ML Panel",
            "T      Toggle Tables",
            "C      AI Chat 🤖",
        ]

        panel_width = 180
        panel_height = len(legend_lines) * 22 + 15
        panel_x = LEADERBOARD_PADDING
        panel_y = LEADERBOARD_PADDING

        # Draw panel background
        bg_rect = arcade.XYWH(
            panel_x + panel_width / 2,
            panel_y + panel_height / 2,
            panel_width,
            panel_height
        )
        arcade.draw_rect_filled(bg_rect, PANEL_BG_COLOR)
        arcade.draw_rect_outline(bg_rect, PANEL_BORDER_COLOR, 2)

        for i, line in enumerate(legend_lines):
            text_color = arcade.color.WHITE if i == 0 else arcade.color.LIGHT_GRAY
            font_size = 12 if i == 0 else 10
            arcade.Text(
                line,
                panel_x + 10,
                panel_y + panel_height - 15 - (i * 22),
                text_color,
                font_size,
                bold=(i == 0)
            ).draw()

    def _draw_selected_driver_info(self, frame):
        """Draw selected driver information panel."""
        if not self.selected_driver or self.selected_driver not in frame["drivers"]:
            return

        driver_pos = frame["drivers"][self.selected_driver]
        driver_color = self.driver_colors.get(self.selected_driver, arcade.color.GRAY)

        # Position below HUD panel
        info_x = LEADERBOARD_PADDING
        info_y = self.height - LEADERBOARD_PADDING - HUD_PANEL_HEIGHT - 20
        box_width = 260
        box_height = 200

        # Background box
        bg_rect = arcade.XYWH(
            info_x + box_width / 2,
            info_y - box_height / 2,
            box_width,
            box_height
        )
        arcade.draw_rect_filled(bg_rect, PANEL_BG_COLOR)
        arcade.draw_rect_outline(bg_rect, driver_color, 3)

        # Driver name header
        header_rect = arcade.XYWH(
            info_x + box_width / 2,
            info_y - 5,
            box_width,
            35
        )
        arcade.draw_rect_filled(header_rect, driver_color)

        # Team color bar
        arcade.draw_rect_filled(arcade.XYWH(info_x + 8, info_y - 5, 6, 25), (255, 255, 255))

        arcade.Text(
            f"  {self.selected_driver}",
            info_x + 15,
            info_y - 5,
            arcade.color.WHITE,
            18,
            bold=True,
            anchor_x="left", anchor_y="center"
        ).draw()

        # Driver stats
        stat_y = info_y - 40
        speed = driver_pos.get('speed', 0)
        gear = driver_pos.get('gear', 0)
        drs_value = driver_pos.get('drs', 0)
        lap = driver_pos.get('lap', 1)
        tyre = get_tyre_compound_str(driver_pos.get('tyre', 1))

        # DRS status
        if drs_value in [0, 1]:
            drs_status = ("Off", arcade.color.GRAY)
        elif drs_value == 8:
            drs_status = ("Ready", arcade.color.YELLOW)
        elif drs_value in [10, 12, 14]:
            drs_status = ("ACTIVE", arcade.color.GREEN)
        else:
            drs_status = ("--", arcade.color.GRAY)

        stats = [
            ("🏎 Speed", f"{speed:.0f} km/h", arcade.color.WHITE),
            ("⚙ Gear", str(gear), arcade.color.WHITE),
            ("📡 DRS", drs_status[0], drs_status[1]),
            ("🔄 Lap", str(lap), arcade.color.WHITE),
            ("🛞 Tyre", tyre, self._get_tyre_color(tyre)),
        ]

        for i, (label, value, color) in enumerate(stats):
            y_pos = stat_y - (i * 28)
            arcade.Text(label, info_x + 15, y_pos,
                        arcade.color.LIGHT_GRAY, 12, anchor_y="center").draw()
            arcade.Text(value, info_x + box_width - 15, y_pos,
                        color, 13, bold=True, anchor_x="right", anchor_y="center").draw()

        # ML Prediction
        prediction = self.ml_predictor.predict(frame, self.selected_driver) if self.ml_trained else None
        if prediction:
            trend = prediction['trend']
            trend_color = (arcade.color.GREEN if trend == 'improving' else
                           arcade.color.RED if trend == 'declining' else arcade.color.GRAY)
            pred_text = f"→ P{prediction['predicted_position']:.0f}"
            arcade.Text("🤖 Prediction", info_x + 15, stat_y - 140,
                        arcade.color.CYAN, 12, anchor_y="center").draw()
            arcade.Text(pred_text, info_x + box_width - 15, stat_y - 140,
                        trend_color, 13, bold=True, anchor_x="right", anchor_y="center").draw()

    def _get_tyre_color(self, tyre_name):
        """Get color for tyre compound."""
        colors = {
            "SOFT": arcade.color.RED,
            "MEDIUM": arcade.color.YELLOW,
            "HARD": arcade.color.WHITE,
            "INTERMEDIATE": arcade.color.GREEN,
            "WET": arcade.color.BLUE,
        }
        return colors.get(tyre_name.upper(), arcade.color.GRAY)

    def _draw_ml_panel(self, frame):
        """Draw ML prediction panel at bottom right."""
        if not self.show_ml_panel:
            return

        # Position at bottom right, above controls
        panel_x = self.width - ML_PANEL_WIDTH - LEADERBOARD_PADDING
        panel_y = LEADERBOARD_PADDING

        # Panel background
        bg_rect = arcade.XYWH(
            panel_x + ML_PANEL_WIDTH / 2,
            panel_y + ML_PANEL_HEIGHT / 2,
            ML_PANEL_WIDTH,
            ML_PANEL_HEIGHT
        )
        arcade.draw_rect_filled(bg_rect, PANEL_BG_COLOR)
        arcade.draw_rect_outline(bg_rect, arcade.color.CYAN, 2)

        # Panel header
        header_rect = arcade.XYWH(
            panel_x + ML_PANEL_WIDTH / 2,
            panel_y + ML_PANEL_HEIGHT - 18,
            ML_PANEL_WIDTH,
            36
        )
        arcade.draw_rect_filled(header_rect, (0, 80, 100, 200))

        arcade.Text(
            "🤖 ML RACE INSIGHTS",
            panel_x + 15,
            panel_y + ML_PANEL_HEIGHT - 18,
            arcade.color.CYAN,
            14,
            bold=True,
            anchor_x="left", anchor_y="center"
        ).draw()

        # Training status indicator
        status_color = arcade.color.GREEN if self.ml_trained else arcade.color.YELLOW
        status_text = "● ACTIVE" if self.ml_trained else "● TRAINING..."
        arcade.Text(
            status_text,
            panel_x + ML_PANEL_WIDTH - 15,
            panel_y + ML_PANEL_HEIGHT - 18,
            status_color,
            11,
            anchor_x="right", anchor_y="center"
        ).draw()

        # Update insights periodically
        if int(self.frame_index) % 50 == 0 and self.ml_trained:
            if self.mode == 'predicted':
                # Use predicted mode insights generator
                self.ml_insights = self._generate_predicted_insights(frame)
            else:
                # Use ML predictor for historical mode
                self.ml_insights = self.ml_predictor.get_race_insights(frame)

        # Draw insights with icons
        insight_y = panel_y + ML_PANEL_HEIGHT - 50
        for i, insight in enumerate(self.ml_insights[:4]):
            # Truncate if too long using constant
            if len(insight) < ML_INSIGHT_MAX_LENGTH:
                display_text = insight
            else:
                display_text = insight[:ML_INSIGHT_MAX_LENGTH - 3] + "..."
            arcade.Text(
                display_text,
                panel_x + 15,
                insight_y - (i * 32),
                arcade.color.WHITE,
                12,
                anchor_x="left", anchor_y="center"
            ).draw()

    def _draw_chat_panel(self, frame):
        """Draw the AI chat panel."""
        if not self.show_chat_panel:
            return

        # Center the chat panel
        panel_x = (self.width - CHAT_PANEL_WIDTH) / 2
        panel_y = (self.height - CHAT_PANEL_HEIGHT) / 2

        # Draw panel background
        bg_rect = arcade.XYWH(
            self.width / 2,
            self.height / 2,
            CHAT_PANEL_WIDTH,
            CHAT_PANEL_HEIGHT
        )
        arcade.draw_rect_filled(bg_rect, (20, 20, 35, 240))
        arcade.draw_rect_outline(bg_rect, arcade.color.CYAN, 3)

        # Draw header
        header_rect = arcade.XYWH(
            self.width / 2,
            panel_y + CHAT_PANEL_HEIGHT - 25,
            CHAT_PANEL_WIDTH,
            50
        )
        arcade.draw_rect_filled(header_rect, (0, 80, 100, 220))

        arcade.Text(
            "🤖 AI Race Analyst - Ask anything about F1!",
            self.width / 2,
            panel_y + CHAT_PANEL_HEIGHT - 25,
            arcade.color.CYAN,
            16,
            bold=True,
            anchor_x="center", anchor_y="center"
        ).draw()

        # Draw chat messages
        msg_y = panel_y + CHAT_PANEL_HEIGHT - 70
        for i, msg in enumerate(self.chat_messages[-6:]):  # Show last 6 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Truncate long messages
            max_chars = 60
            if len(content) > max_chars:
                content = content[:max_chars - 3] + "..."
            
            if role == "user":
                prefix = "You: "
                color = arcade.color.LIGHT_BLUE
            else:
                prefix = "AI: "
                color = arcade.color.LIGHT_GREEN

            arcade.Text(
                prefix + content,
                panel_x + 20,
                msg_y - (i * 40),
                color,
                12,
                anchor_x="left", anchor_y="center"
            ).draw()

        # Draw input box
        input_y = panel_y + 35
        input_rect = arcade.XYWH(
            self.width / 2,
            input_y,
            CHAT_PANEL_WIDTH - 40,
            CHAT_INPUT_HEIGHT
        )
        
        # Highlight input box when active
        input_bg_color = (50, 50, 70) if self.chat_input_active else (30, 30, 50)
        arcade.draw_rect_filled(input_rect, input_bg_color)
        arcade.draw_rect_outline(input_rect, arcade.color.WHITE if self.chat_input_active else arcade.color.GRAY, 2)

        # Draw input text or placeholder
        if self.chat_input:
            display_text = self.chat_input
            if len(display_text) > 45:
                display_text = "..." + display_text[-42:]
            text_color = arcade.color.WHITE
        else:
            display_text = "Type your question and press Enter..."
            text_color = arcade.color.GRAY

        arcade.Text(
            display_text,
            panel_x + 25,
            input_y,
            text_color,
            13,
            anchor_x="left", anchor_y="center"
        ).draw()

        # Draw close instruction
        arcade.Text(
            "Press C to close | ESC to cancel input",
            self.width / 2,
            panel_y + 10,
            arcade.color.LIGHT_GRAY,
            11,
            anchor_x="center", anchor_y="center"
        ).draw()

        # Show quick tips if no messages yet
        if not self.chat_messages:
            tips = self.ai_chat.get_quick_tips()
            tip_y = panel_y + CHAT_PANEL_HEIGHT - 120
            for i, tip in enumerate(tips[:4]):
                arcade.Text(
                    tip,
                    panel_x + 30,
                    tip_y - (i * 30),
                    arcade.color.LIGHT_GRAY,
                    11,
                    anchor_x="left", anchor_y="center"
                ).draw()

    def _send_chat_message(self):
        """Send the current chat input to the AI."""
        if not self.chat_input.strip():
            return

        question = self.chat_input.strip()
        self.chat_input = ""

        # Add user message to history
        self.chat_messages.append({"role": "user", "content": question})

        # Get current frame for context
        idx = min(int(self.frame_index), self.n_frames - 1)
        frame = self.frames[idx]

        # Update AI context with current standings
        standings = {}
        for code, data in frame['drivers'].items():
            standings[code] = data.get('position', 99)
        self.ai_chat.set_race_context(self.race_info, self.drivers, standings)

        # Get AI response
        response = self.ai_chat.ask(question)
        self.chat_messages.append({"role": "assistant", "content": response})

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
        elif symbol == arcade.key.C:
            # Toggle chat panel (only when not actively typing)
            if not self.chat_input_active:
                self.show_chat_panel = not self.show_chat_panel
                if self.show_chat_panel:
                    self.chat_input_active = True
                    self.paused = True  # Pause when chat is open
                else:
                    self.chat_input_active = False
            # Note: 'c' character input is handled by on_text
        elif symbol == arcade.key.ESCAPE:
            if self.show_chat_panel:
                # Close chat panel
                self.show_chat_panel = False
                self.chat_input_active = False
                self.chat_input = ""
        elif symbol == arcade.key.ENTER or symbol == arcade.key.RETURN:
            if self.chat_input_active and self.chat_input.strip():
                self._send_chat_message()
        elif symbol == arcade.key.BACKSPACE:
            if self.chat_input_active and self.chat_input:
                self.chat_input = self.chat_input[:-1]

    def on_text(self, text: str):
        """Handle text input for chat."""
        if self.chat_input_active and self.show_chat_panel:
            # Allow alphanumeric, spaces, and common punctuation for chat input
            # Exclude control characters and newlines for single-line input
            allowed_chars = set(' .,!?-\'":;()[]{}@#$%&*+=/<>~`')
            if len(self.chat_input) < 200:
                if text.isalnum() or text in allowed_chars:
                    self.chat_input += text

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int):
        """Handle mouse click for driver selection."""
        # If chat panel is open, check if clicking in input area
        if self.show_chat_panel:
            panel_x = (self.width - CHAT_PANEL_WIDTH) / 2
            panel_y = (self.height - CHAT_PANEL_HEIGHT) / 2
            input_y = panel_y + 35
            
            # Check if click is in input box area
            if (panel_x + 20 <= x <= panel_x + CHAT_PANEL_WIDTH - 20 and
                input_y - 20 <= y <= input_y + 20):
                self.chat_input_active = True
                return
            else:
                self.chat_input_active = False
        
        new_selection = None
        for code, left, bottom, right, top in self.leaderboard_rects:
            if left <= x <= right and bottom <= y <= top:
                new_selection = code
                break

        if new_selection == self.selected_driver:
            self.selected_driver = None
        else:
            self.selected_driver = new_selection


def run_arcade_replay(frames=None, track_statuses=None, example_lap=None, drivers=None, title="F1 Race Replay",
                      playback_speed=1.0, driver_colors=None, predictions=None,
                      mode='historical', race_info=None,
                      driver_data_array=None, frame_metadata=None, driver_codes=None):
    """Run the F1 replay visualization.

    Supports both legacy frame format and optimized NumPy arrays for better performance.

    Args:
        frames: Race telemetry frames (legacy format, optional if using NumPy arrays)
        track_statuses: Track status data
        example_lap: Example lap for track geometry
        drivers: List of driver codes (used with legacy format)
        title: Window title
        playback_speed: Initial playback speed multiplier
        driver_colors: Dictionary mapping driver codes to RGB colors
        predictions: Optional dictionary of ML predictions
        mode: 'historical' for replays, 'predicted' for future race predictions
        race_info: Dictionary with race information (year, gp, etc.)
        driver_data_array: NumPy 3D array (n_frames, n_drivers, n_fields) - optimized format
        frame_metadata: NumPy 2D array (n_frames, 2) with [time, leader_lap] - optimized format
        driver_codes: List of driver codes (used with NumPy arrays)
    """
    F1ReplayWindow(
        frames=frames,
        track_statuses=track_statuses,
        example_lap=example_lap,
        drivers=drivers or driver_codes,
        playback_speed=playback_speed,
        driver_colors=driver_colors,
        title=title,
        predictions=predictions,
        mode=mode,
        race_info=race_info,
        driver_data_array=driver_data_array,
        frame_metadata=frame_metadata,
        driver_codes=driver_codes
    )
    arcade.run()
