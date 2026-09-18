"""
F1 Race Replay with Arcade graphics library.
Based on f1-race-replay project by Tom Shaw.
Extended with ML prediction capabilities and AI chat.

Optimized for performance with batch rendering and NumPy data access.
"""

import os
import sys
import arcade
import numpy as np
from typing import Optional, Union, List
from src.f1_data import (
    FPS, INTERPOLATION_FACTOR, FIELD_X, FIELD_Y, FIELD_DIST, FIELD_REL_DIST, FIELD_LAP,
    FIELD_TYRE, FIELD_SPEED, FIELD_GEAR, FIELD_DRS, FIELD_POSITION
)
from src.ml_predictor import RaceTrendPredictor
from src.lib.tyres import get_tyre_compound_str
from src.dashboard.prediction_overlay import PredictionOverlay
from src.ai_chat import F1AIChat

# Fix for macOS recursion error in pyglet/cocoapy
if sys.platform == 'darwin':
    import resource
    # Increase recursion limit for macOS
    sys.setrecursionlimit(50000)
    # Also increase stack size if possible
    try:
        resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except:
        pass

# Default screen dimensions (1080p for better compatibility)
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
SCREEN_TITLE = "F1 Race Replay with ML Prediction"

# UI Layout Constants
LEADERBOARD_WIDTH = 360  # Increased from 280 to fit interval and time columns
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

# Car rendering constants
CAR_RADIUS = 8  # Radius of car circles on track
CAR_OUTLINE_WIDTH = 2  # Thickness of the white ring drawn around each car

# --- Animation / frame pacing -------------------------------------------------
# Playback is driven by a floating point ``frame_index``: its integer part picks
# a stored telemetry frame and its fractional part blends towards the next one.
# How smooth the replay looks therefore depends on two things - how evenly that
# index advances, and how cheap a single rendered frame is.

# Arcade schedules both on_update() and on_draw() at this rate. Pinning the draw
# rate as well as the update rate keeps the two loops in step; otherwise arcade
# falls back to drawing "as fast as possible", which produces uneven pacing.
TARGET_FPS = 60

# A single slow tick (garbage collection, a window drag, a slow ML call) makes
# arcade report the whole stall as one huge delta_time. Advancing playback by
# that raw value teleports the cars across the track, so cap what one update may
# consume - the replay falls slightly behind real time instead of jumping.
MAX_FRAME_DELTA = 1.0 / 15.0  # seconds

# Even on a healthy 60 Hz display the deltas wobble (16.1 ms, 17.4 ms, 15.9 ms).
# Advancing by the raw value turns that wobble into visible shimmer, so we
# advance by an exponential moving average instead. The average converges to the
# true frame time, so playback speed stays accurate - only the jitter is lost.
DELTA_SMOOTHING = 0.12  # 0.0 = never adapt, 1.0 = no smoothing at all

# How much race time the left/right arrow keys skip. Expressed in seconds
# because the number of stored frames per second depends on INTERPOLATION_FACTOR.
SEEK_SECONDS = 3.0

# Expensive machine-learning work is thrown out of on_draw() and throttled by
# wall-clock time here, so a slow model call can never stall a rendered frame.
ML_PREDICTION_INTERVAL = 1.0  # seconds between live position predictions
ML_INSIGHT_INTERVAL = 2.0     # seconds between insight-text refreshes

# Telemetry columns holding a continuous quantity may be blended between two
# stored frames. Everything else is a discrete state where blending would
# produce nonsense such as "lap 12.4" or a fractional tyre compound id, so those
# columns are copied from the current frame unchanged.
DISCRETE_FIELDS = (
    FIELD_REL_DIST, FIELD_LAP, FIELD_TYRE, FIELD_GEAR, FIELD_DRS, FIELD_POSITION,
)

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
        # vsync stops the screen tearing; pinning update_rate *and* draw_rate keeps
        # the simulation step and the render step on the same even cadence.
        super().__init__(
            SCREEN_WIDTH, SCREEN_HEIGHT, title,
            resizable=True,
            vsync=True,
            update_rate=1 / TARGET_FPS,
            draw_rate=1 / TARGET_FPS,
        )

        self.track_statuses = track_statuses
        self.playback_speed = playback_speed
        self.driver_colors = driver_colors or {}
        self.paused = False
        self._tyre_textures = {}

        # --- Animation state -------------------------------------------------
        # Fractional on purpose: int(frame_index) selects the stored frame and the
        # remainder is the blend factor towards the next one (see _get_frame_state).
        self.frame_index = 0.0
        # Low-pass filtered frame delta used to advance frame_index (see on_update).
        self._smoothed_delta = 1.0 / TARGET_FPS
        # Wall-clock seconds since the window opened; drives the throttles below.
        self._elapsed_time = 0.0
        self._next_prediction_time = 0.0
        self._next_insight_time = 0.0
        # Memoised _get_frame_state() result, so the six panels drawn in a single
        # frame share one computation - and therefore agree on every position.
        self._frame_state_cache_key = None
        self._frame_state_cache = None
        # Long-lived arcade.Text objects keyed by call site (see _get_text()).
        self._text_cache = {}
        self._text_colors = {}
        # Cached ML prediction for the selected driver, refreshed on a timer.
        self._selected_driver_prediction = None

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
            # Pre-built index array for restoring the non-blendable columns after
            # a vectorised interpolation (see _build_state_from_arrays).
            self._discrete_field_idx = np.array(DISCRETE_FIELDS, dtype=np.intp)
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
        self._track_shapes: Optional[arcade.shape_list.ShapeElementList] = None
        self._last_track_status = None  # Track the last status to detect changes

        # Batch rendering - Car sprites (body + outline, see _init_car_sprites)
        self._car_sprites: Optional[arcade.SpriteList] = None
        self._car_outline_sprites: Optional[arcade.SpriteList] = None
        self._car_sprite_map: dict = {}  # Maps driver code to (outline, body)
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
            # Nothing to train on in prediction mode, but the panels treat this
            # flag as "outputs are usable"; insights come from
            # _generate_predicted_insights() instead of the model.
            self.ml_trained = True

    def _init_car_sprites(self):
        """Create the sprites used to draw the cars.

        Each car is two stacked sprites: a slightly larger white circle behind a
        team-coloured body. Drawing the ring as a sprite means the whole field
        costs two batched SpriteList draw calls per frame, instead of one
        immediate-mode draw_circle_outline() call per car per frame (twenty
        separate GPU round trips that used to run on top of the batch).
        """
        self._car_outline_sprites = arcade.SpriteList()
        self._car_sprites = arcade.SpriteList()
        self._car_sprite_map = {}

        for code in self.drivers:
            color = self.driver_colors.get(code, arcade.color.WHITE)
            outline = arcade.SpriteCircle(CAR_RADIUS + CAR_OUTLINE_WIDTH, arcade.color.WHITE)
            body = arcade.SpriteCircle(CAR_RADIUS, color)
            # Start hidden; _update_car_sprites() reveals and positions them.
            outline.visible = False
            body.visible = False
            self._car_outline_sprites.append(outline)
            self._car_sprites.append(body)
            self._car_sprite_map[code] = (outline, body)

    def _get_text(self, key, text, x, y, color, font_size=12, bold=False,
                  anchor_x="left", anchor_y="baseline"):
        """Return a reusable arcade.Text object for one call site.

        Constructing an arcade.Text lays the string out glyph by glyph and
        uploads the result to the GPU. Rebuilding every label on every frame -
        the leaderboard alone is about a hundred of them - was by far the most
        expensive thing this window did, and it is what made the replay feel
        choppy even when very little was moving.

        Here each call site keeps one long-lived Text object and we only write
        back the attributes that actually changed, which costs a few comparisons.

        Args:
            key: Unique, stable identifier for the call site (e.g. "hud.lap").
            text: The string to display.
            x, y: Screen position.
            color: Text colour as an RGB or RGBA tuple.
            font_size, bold, anchor_x, anchor_y: Applied once, when the object is
                first created - these cannot change for a given key.

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

        # Every assignment below invalidates arcade's cached layout even when the
        # value is unchanged, so each one is guarded by a comparison.
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

    def _build_track_shapes(self, track_color):
        """Build static track geometry as ShapeElementList for batch rendering.
        
        Only regenerate when track status (color) changes.
        Uses adaptive line width based on track scale for better visual quality.
        """
        self._track_shapes = arcade.shape_list.ShapeElementList()
        
        # Adaptive line width based on world scale (looks better at different zoom levels)
        line_width = max(2, min(4, int(self.world_scale * 0.5)))
        
        # Create line strips for inner and outer track edges
        if len(self.screen_inner_points) > 1:
            inner_line = arcade.shape_list.create_line_strip(
                self.screen_inner_points, track_color, line_width
            )
            self._track_shapes.append(inner_line)
        
        if len(self.screen_outer_points) > 1:
            outer_line = arcade.shape_list.create_line_strip(
                self.screen_outer_points, track_color, line_width
            )
            self._track_shapes.append(outer_line)
        
        self._last_track_status = track_color

    def _get_frame_state(self, frame_index: float) -> dict:
        """Build the race state for a *fractional* frame index.

        ``frame_index`` usually sits between two stored telemetry frames. Values
        that describe a continuous quantity (track position, distance covered,
        speed, the race clock) are linearly blended towards the next frame so the
        cars glide; discrete values (lap number, tyre compound, gear, DRS, race
        position) are taken from the current frame, because a blended "lap 12.4"
        or a fractional tyre id is meaningless.

        The result is memoised per index. Every panel in a rendered frame then
        shares one computation *and* one set of positions, so the car, its label
        and its battle marker can never disagree about where it is.

        Args:
            frame_index: Fractional index into the telemetry frames.

        Returns:
            dict with ``t`` (race seconds), ``lap`` (leader lap) and ``drivers``
            (driver code -> telemetry dict), matching the legacy frame layout.
        """
        if self._frame_state_cache_key == frame_index:
            return self._frame_state_cache

        idx = max(0, min(int(frame_index), self.n_frames - 1))
        next_idx = min(idx + 1, self.n_frames - 1)
        # No next frame to blend into on the very last frame of the replay.
        blend = (frame_index - idx) if next_idx != idx else 0.0

        if self.use_numpy_arrays:
            state = self._build_state_from_arrays(idx, next_idx, blend)
        else:
            state = self._build_state_from_frames(idx, next_idx, blend)

        self._frame_state_cache_key = frame_index
        self._frame_state_cache = state
        return state

    def _build_state_from_arrays(self, idx: int, next_idx: int, blend: float) -> dict:
        """Interpolate the optimised NumPy telemetry arrays (fast path).

        The blend is a single vectorised operation over the whole
        ``(n_drivers, n_fields)`` slice rather than a Python loop over every
        driver and every field, and ``.tolist()`` then converts the block to
        Python floats in one C-level pass.
        """
        current = self.driver_data_array[idx]

        if blend > 0.0:
            values = current + (self.driver_data_array[next_idx] - current) * blend
            # Undo the blend for the columns that must stay discrete.
            values[:, self._discrete_field_idx] = current[:, self._discrete_field_idx]
        else:
            values = current

        rows = values.tolist()

        drivers = {}
        for driver_idx, code in enumerate(self.driver_codes):
            row = rows[driver_idx]
            drivers[code] = {
                "x": row[FIELD_X],
                "y": row[FIELD_Y],
                "dist": row[FIELD_DIST],
                "rel_dist": row[FIELD_REL_DIST],
                "lap": int(round(row[FIELD_LAP])),
                "tyre": int(row[FIELD_TYRE]),
                "position": int(row[FIELD_POSITION]),
                "speed": row[FIELD_SPEED],
                "gear": int(row[FIELD_GEAR]),
                "drs": int(row[FIELD_DRS]),
            }

        # Blend the race clock too, otherwise the HUD timer ticks in visible steps.
        race_time = float(self.frame_metadata[idx, 0])
        if blend > 0.0:
            race_time += (float(self.frame_metadata[next_idx, 0]) - race_time) * blend

        return {
            "t": race_time,
            "lap": int(self.frame_metadata[idx, 1]),
            "drivers": drivers,
        }

    def _build_state_from_frames(self, idx: int, next_idx: int, blend: float) -> dict:
        """Interpolate the legacy list-of-dicts frame format.

        Kept so callers that still pass ``frames=`` keep working; the NumPy path
        above is used whenever the optimised arrays are supplied.
        """
        current = self.frames[idx]
        if blend <= 0.0:
            return current

        next_frame = self.frames[next_idx]
        drivers = {}

        for code, pos in current["drivers"].items():
            next_pos = next_frame["drivers"].get(code)
            if next_pos is None:
                # Driver missing from the next frame - nothing sensible to blend to.
                drivers[code] = pos
                continue

            blended = dict(pos)
            # Only the continuous values move; see _get_frame_state() for why.
            for field in ("x", "y", "dist", "speed"):
                start = pos.get(field, 0.0)
                blended[field] = start + (next_pos.get(field, start) - start) * blend
            drivers[code] = blended

        return {
            "t": current["t"] + (next_frame["t"] - current["t"]) * blend,
            "lap": current.get("lap", 1),
            "drivers": drivers,
        }

    def _update_car_sprites(self, frame: dict) -> dict:
        """Move every car sprite to its screen position for this frame.

        ``frame`` is already interpolated by _get_frame_state(), so this only has
        to convert world coordinates to screen coordinates. The scale/offset are
        copied into locals first because this runs once per driver per frame and
        attribute lookups add up in a hot loop.

        Args:
            frame: Interpolated race state from _get_frame_state().

        Returns:
            Mapping of driver code -> (screen_x, screen_y) for the visible cars,
            reused by the label and overlay passes so they stay exactly in sync.
        """
        screen_positions = {}
        scale, tx, ty = self.world_scale, self.tx, self.ty

        for code, pos in frame["drivers"].items():
            sprites = self._car_sprite_map.get(code)
            if sprites is None:
                continue

            outline, body = sprites

            # rel_dist == 1 marks a retired car: hide it rather than parking it
            # on the track for the rest of the replay.
            if pos.get("rel_dist", 0) == 1:
                outline.visible = False
                body.visible = False
                continue

            sx = scale * pos["x"] + tx
            sy = scale * pos["y"] + ty
            outline.position = (sx, sy)
            body.position = (sx, sy)
            outline.visible = True
            body.visible = True
            screen_positions[code] = (sx, sy)

        return screen_positions

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
        # Silently train the model
        if self.use_numpy_arrays:
            # ML predictor can accept NumPy arrays directly
            if self.ml_predictor.train_from_numpy(
                self.driver_data_array, self.frame_metadata, self.driver_codes
            ):
                self.ml_trained = True
                self.ml_insights = ["ML model ready"]
            else:
                self.ml_insights = ["ML training failed - insufficient data"]
        else:
            if self.ml_predictor.train(self.frames, self.drivers):
                self.ml_trained = True
                self.ml_insights = ["ML model ready"]
            else:
                self.ml_insights = ["ML training failed - insufficient data"]

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

        # 2. Build the interpolated race state once and share it with every panel
        # below, so the cars, the leaderboard and the HUD all describe the exact
        # same instant of the race.
        frame = self._get_frame_state(self.frame_index)
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

        # 4. Draw Cars - two batched SpriteList calls (white ring, then body)
        # instead of per-car immediate-mode circles.
        # The returned screen coordinates are shared with the label and overlay
        # passes below, so nothing can drift out of sync with the sprites.
        screen_positions = self._update_car_sprites(frame)
        self._car_outline_sprites.draw()
        self._car_sprites.draw()

        # 5. Draw name tags for the podium places and the selected driver. They
        # reuse the screen coordinates computed above, so a tag can never lag a
        # pixel behind the car it belongs to.
        self._draw_car_labels(frame, screen_positions)

        # --- UI ELEMENTS ---
        self._draw_hud(frame, current_time, current_track_status)
        self._draw_leaderboard(frame)
        self._draw_prediction_overlay(screen_positions)
        self._draw_controls_legend()
        self._draw_selected_driver_info(frame)
        self._draw_ml_panel(frame)

        # Draw tables view (on top of everything when active)
        self.prediction_overlay.draw_tables_view(self.width, self.height, frame)

        # Draw chat panel (on top of everything when active)
        self._draw_chat_panel(frame)

    def _draw_car_labels(self, frame, screen_positions):
        """Draw the driver-code tags that follow the top three and the selection.

        Args:
            frame: Interpolated race state from _get_frame_state().
            screen_positions: driver code -> (x, y) from _update_car_sprites().
        """
        for code, (sx, sy) in screen_positions.items():
            position = frame["drivers"][code].get("position", 99)
            if code != self.selected_driver and position > 3:
                continue

            label_y = sy + 18
            # Small dark plate behind the text so it stays readable over the track.
            arcade.draw_rect_filled(arcade.XYWH(sx, label_y, 28, 14), (0, 0, 0, 180))
            self._get_text(
                f"car.label.{code}", code, sx, label_y,
                arcade.color.WHITE, 9, bold=True,
                anchor_x="center", anchor_y="center",
            ).draw()

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
            self._get_text("hud.banner", banner_text,
                           self.width / 2, self.height - 15,
                           arcade.color.CYAN, 18, bold=True,
                           anchor_x="center", anchor_y="top").draw()

        # HUD content
        text_x = panel_x + 15
        text_y = self.height - LEADERBOARD_PADDING - 20
        line_offset = 0

        # Lap counter with larger font
        self._get_text("hud.lap", f"LAP {leader_lap}",
                       text_x, text_y - line_offset,
                       arcade.color.WHITE, 28, bold=True, anchor_y="top").draw()
        line_offset += HUD_SECTION_GAP

        # Race time
        self._get_text("hud.time", f"⏱ {time_str}",
                       text_x, text_y - line_offset,
                       arcade.color.LIGHT_GRAY, 18, anchor_y="top").draw()
        line_offset += HUD_LINE_HEIGHT

        # Playback speed
        speed_color = arcade.color.GREEN if self.playback_speed > 1 else (
            arcade.color.YELLOW if self.playback_speed < 1 else arcade.color.WHITE
        )
        self._get_text("hud.speed", f"▶ {self.playback_speed}x",
                       text_x, text_y - line_offset,
                       speed_color, 16, anchor_y="top").draw()
        line_offset += HUD_LINE_HEIGHT

        # Pause indicator
        if self.paused:
            self._get_text("hud.paused", "⏸ PAUSED",
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
            self._get_text("hud.flag", text,
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
        self._get_text("lb.header", "🏁 LIVE STANDINGS",
                       leaderboard_x + 10, leaderboard_y - 12,
                       arcade.color.WHITE, 16, bold=True,
                       anchor_x="left", anchor_y="top").draw()

        # Race time. The key is "t" - reading "time" always returned the default
        # of 0, so this clock was frozen at 00:00 for the whole replay.
        race_time = frame.get("t", 0)
        time_mins = int(race_time // 60)
        time_secs = int(race_time % 60)
        time_text = f"⏱ {time_mins:02d}:{time_secs:02d}"
        self._get_text("lb.time", time_text,
                       leaderboard_x + LEADERBOARD_WIDTH - 10, leaderboard_y - 12,
                       arcade.color.YELLOW, 14, bold=True,
                       anchor_x="right", anchor_y="top").draw()

        # Prepare driver list
        driver_list = []
        for code, pos in frame["drivers"].items():
            color = self.driver_colors.get(code, arcade.color.WHITE)
            driver_list.append((code, color, pos))

        # Sort by distance (race position)
        driver_list.sort(key=lambda x: x[2].get("dist", 999), reverse=True)

        # Get leader distance and speed for gap/interval calculation
        if driver_list:
            leader_dist = driver_list[0][2].get("dist", 0)
            leader_speed = driver_list[0][2].get("speed", 1)

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
            # Cache keys are per row index, not per driver: a row keeps its place
            # on screen while the driver occupying it changes.
            self._get_text(f"lb.pos.{i}", str(current_pos),
                           left_x + 12, top_y - row_height / 2 + 2,
                           arcade.color.BLACK if current_pos <= 3 else arcade.color.WHITE,
                           11, bold=True, anchor_x="center", anchor_y="center").draw()

            # Driver code with team color indicator
            arcade.draw_rect_filled(arcade.XYWH(left_x + 30, top_y - row_height / 2 + 2, 4, 16), color)

            # Check if OUT
            is_out = pos.get("rel_dist", 0) == 1
            text_color = arcade.color.GRAY if is_out else arcade.color.WHITE

            driver_text = f"{code}"
            self._get_text(f"lb.code.{i}", driver_text,
                           left_x + 40, top_y - row_height / 2 + 2,
                           text_color, 13, bold=True,
                           anchor_x="left", anchor_y="center").draw()

            # Calculate interval (to car ahead) and gap (to leader)
            if is_out:
                # Show OUT status. Separate cache keys per variant because the
                # font size and weight differ and are fixed at construction time.
                self._get_text(f"lb.out.{i}", "OUT",
                               right_x - 105, top_y - row_height / 2 + 2,
                               arcade.color.RED, 10, bold=True,
                               anchor_x="right", anchor_y="center").draw()
            elif i == 0:
                # Leader - show "Leader"
                self._get_text(f"lb.leader.{i}", "Leader",
                               right_x - 105, top_y - row_height / 2 + 2,
                               arcade.color.GREEN, 9,
                               anchor_x="right", anchor_y="center").draw()
            else:
                # Calculate interval to car ahead
                car_ahead_dist = driver_list[i-1][2].get("dist", 0)
                current_dist = pos.get("dist", 0)
                interval_dist = car_ahead_dist - current_dist
                
                # Estimate time interval based on average speed (approximate)
                avg_speed = (driver_list[i-1][2].get("speed", 1) + pos.get("speed", 1)) / 2
                if avg_speed > 1:  # Avoid division by zero
                    interval_time = (interval_dist / 1000) / (avg_speed / 3600)  # Convert to seconds
                    interval_text = f"+{interval_time:.1f}s"
                else:
                    interval_text = f"+{interval_dist / 1000:.2f}km"
                
                self._get_text(f"lb.interval.{i}", interval_text,
                               right_x - 105, top_y - row_height / 2 + 2,
                               arcade.color.LIGHT_YELLOW, 9,
                               anchor_x="right", anchor_y="center").draw()

                # Gap to leader
                gap = leader_dist - current_dist
                if leader_speed > 1:
                    gap_time = (gap / 1000) / (leader_speed / 3600)
                    gap_text = f"+{gap_time:.1f}s"
                else:
                    gap_text = f"+{gap / 1000:.1f}km"
                
                self._get_text(f"lb.gap.{i}", gap_text,
                               right_x - 48, top_y - row_height / 2 + 2,
                               arcade.color.LIGHT_GRAY, 9,
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

    def _draw_prediction_overlay(self, screen_positions):
        """Draw the battle markers over the cars.

        This is now pure drawing: refreshing the predictions themselves runs an
        ML model, so it was moved to _update_ml_outputs() where it is throttled
        by wall-clock time and can never stall a rendered frame.

        Args:
            screen_positions: driver code -> (x, y) from _update_car_sprites().
        """
        if not self.prediction_overlay.show_overlay:
            return

        for code, (sx, sy) in screen_positions.items():
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
            "R      Restart",
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
            self._get_text(
                f"legend.{i}",
                line,
                panel_x + 10,
                panel_y + panel_height - 15 - (i * 22),
                text_color,
                font_size,
                bold=(i == 0),
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

        self._get_text(
            "driver.name",
            f"  {self.selected_driver}",
            info_x + 15,
            info_y - 5,
            arcade.color.WHITE,
            18,
            bold=True,
            anchor_x="left", anchor_y="center",
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
            self._get_text(f"driver.stat.label.{i}", label, info_x + 15, y_pos,
                           arcade.color.LIGHT_GRAY, 12, anchor_y="center").draw()
            self._get_text(f"driver.stat.value.{i}", value, info_x + box_width - 15, y_pos,
                           color, 13, bold=True,
                           anchor_x="right", anchor_y="center").draw()

        # ML Prediction - taken from the cache refreshed by _update_ml_outputs()
        # rather than re-running the model on every single rendered frame.
        prediction = self._selected_driver_prediction
        if prediction:
            trend = prediction['trend']
            trend_color = (arcade.color.GREEN if trend == 'improving' else
                           arcade.color.RED if trend == 'declining' else arcade.color.GRAY)
            pred_text = f"→ P{prediction['predicted_position']:.0f}"
            self._get_text("driver.pred.label", "🤖 Prediction", info_x + 15, stat_y - 140,
                           arcade.color.CYAN, 12, anchor_y="center").draw()
            self._get_text("driver.pred.value", pred_text, info_x + box_width - 15, stat_y - 140,
                           trend_color, 13, bold=True,
                           anchor_x="right", anchor_y="center").draw()

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

        self._get_text(
            "ml.header",
            "🤖 ML RACE INSIGHTS",
            panel_x + 15,
            panel_y + ML_PANEL_HEIGHT - 18,
            arcade.color.CYAN,
            14,
            bold=True,
            anchor_x="left", anchor_y="center",
        ).draw()

        # Training status indicator
        status_color = arcade.color.GREEN if self.ml_trained else arcade.color.YELLOW
        status_text = "● ACTIVE" if self.ml_trained else "● TRAINING..."
        self._get_text(
            "ml.status",
            status_text,
            panel_x + ML_PANEL_WIDTH - 15,
            panel_y + ML_PANEL_HEIGHT - 18,
            status_color,
            11,
            anchor_x="right", anchor_y="center",
        ).draw()

        # The insight text itself is regenerated in _update_ml_outputs(); this
        # method only renders whatever is currently in self.ml_insights.

        # Draw insights with icons
        insight_y = panel_y + ML_PANEL_HEIGHT - 50
        for i, insight in enumerate(self.ml_insights[:4]):
            # Truncate if too long using constant
            if len(insight) < ML_INSIGHT_MAX_LENGTH:
                display_text = insight
            else:
                display_text = insight[:ML_INSIGHT_MAX_LENGTH - 3] + "..."
            self._get_text(
                f"ml.insight.{i}",
                display_text,
                panel_x + 15,
                insight_y - (i * 32),
                arcade.color.WHITE,
                12,
                anchor_x="left", anchor_y="center",
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

        self._get_text(
            "chat.header",
            "🤖 AI Race Analyst - Ask anything about F1!",
            self.width / 2,
            panel_y + CHAT_PANEL_HEIGHT - 25,
            arcade.color.CYAN,
            16,
            bold=True,
            anchor_x="center", anchor_y="center",
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

            self._get_text(
                f"chat.msg.{i}",
                prefix + content,
                panel_x + 20,
                msg_y - (i * 40),
                color,
                12,
                anchor_x="left", anchor_y="center",
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

        self._get_text(
            "chat.input",
            display_text,
            panel_x + 25,
            input_y,
            text_color,
            13,
            anchor_x="left", anchor_y="center",
        ).draw()

        # Draw close instruction
        self._get_text(
            "chat.hint",
            "Press C to close | ESC to cancel input",
            self.width / 2,
            panel_y + 10,
            arcade.color.LIGHT_GRAY,
            11,
            anchor_x="center", anchor_y="center",
        ).draw()

        # Show quick tips if no messages yet
        if not self.chat_messages:
            tips = self.ai_chat.get_quick_tips()
            tip_y = panel_y + CHAT_PANEL_HEIGHT - 120
            for i, tip in enumerate(tips[:4]):
                self._get_text(
                    f"chat.tip.{i}",
                    tip,
                    panel_x + 30,
                    tip_y - (i * 30),
                    arcade.color.LIGHT_GRAY,
                    11,
                    anchor_x="left", anchor_y="center",
                ).draw()

    def _send_chat_message(self):
        """Send the current chat input to the AI."""
        if not self.chat_input.strip():
            return

        question = self.chat_input.strip()
        self.chat_input = ""

        # Add user message to history
        self.chat_messages.append({"role": "user", "content": question})

        # Get current frame for context. This must go through _get_frame_state()
        # because self.frames is None whenever the optimised NumPy arrays are used.
        frame = self._get_frame_state(self.frame_index)

        # Update AI context with current standings
        standings = {}
        for code, data in frame['drivers'].items():
            standings[code] = data.get('position', 99)
        self.ai_chat.set_race_context(self.race_info, self.drivers, standings)

        # Get AI response
        response = self.ai_chat.ask(question)
        self.chat_messages.append({"role": "assistant", "content": response})

    def _update_ml_outputs(self):
        """Refresh the machine-learning outputs on a wall-clock schedule.

        These calls run scikit-learn models and rebuild formatted strings. They
        used to sit inside on_draw() and fire on a frame counter, which meant an
        occasional rendered frame took several times as long as its neighbours -
        the classic cause of a visible hitch. Driving them from here, throttled
        by seconds rather than by frames, keeps on_draw() a predictable cost and
        makes the refresh rate independent of the playback speed.
        """
        if not self.ml_trained:
            return

        # Built lazily: if none of the throttles are due we never pay for it.
        frame = None

        # Live position predictions feeding the battle markers. Skipped in
        # predicted mode, where the predictor was never trained on frame data.
        if (self.mode != 'predicted'
                and self.prediction_overlay.show_overlay
                and self._elapsed_time >= self._next_prediction_time):
            self._next_prediction_time = self._elapsed_time + ML_PREDICTION_INTERVAL
            frame = self._get_frame_state(self.frame_index)
            try:
                live_predictions = self.ml_predictor.predict_all_drivers(frame)
            except Exception:
                live_predictions = None
            if live_predictions:
                # External (pre-race) predictions stay as the base layer.
                self.prediction_overlay.update_predictions(
                    {**self.external_predictions, **live_predictions}
                )

        # Insight text shown in the ML panel.
        if self.show_ml_panel and self._elapsed_time >= self._next_insight_time:
            self._next_insight_time = self._elapsed_time + ML_INSIGHT_INTERVAL
            if frame is None:
                frame = self._get_frame_state(self.frame_index)
            try:
                if self.mode == 'predicted':
                    self.ml_insights = self._generate_predicted_insights(frame)
                else:
                    self.ml_insights = self.ml_predictor.get_race_insights(frame)
            except Exception:
                # A failed refresh just leaves the previous insights on screen.
                pass

        # Trend arrow for the driver info panel, on the same schedule.
        if self.mode != 'predicted' and self.selected_driver:
            if frame is None:
                frame = self._get_frame_state(self.frame_index)
            if self.selected_driver in frame["drivers"]:
                try:
                    self._selected_driver_prediction = self.ml_predictor.predict(
                        frame, self.selected_driver
                    )
                except Exception:
                    self._selected_driver_prediction = None

    def on_update(self, delta_time: float):
        """Advance the replay clock.

        Arcade calls this TARGET_FPS times a second. The whole job is turning
        elapsed real seconds into a smooth advance of ``frame_index``.

        Args:
            delta_time: Real seconds since the previous update, as reported by
                arcade. It is neither capped nor evenly spaced, so both are fixed
                here before the value is used.
        """
        # Cap outliers first. After a stall arcade reports the entire gap as one
        # delta; advancing by it would teleport the cars across the track.
        delta_time = min(delta_time, MAX_FRAME_DELTA)

        # Then low-pass filter what is left, so millisecond-scale wobble in the
        # frame times does not show up as shimmer in the car positions. The
        # average converges on the true frame time, so playback speed is
        # unaffected - only the jitter is removed.
        self._smoothed_delta += (delta_time - self._smoothed_delta) * DELTA_SMOOTHING

        # The ML throttles run even while paused so the panels stay live and the
        # work stays out of on_draw().
        self._elapsed_time += delta_time
        self._update_ml_outputs()

        if self.paused:
            return

        # One stored frame is 1 / (FPS * INTERPOLATION_FACTOR) seconds of race
        # time, so this converts smoothed real seconds into stored frames.
        self.frame_index += (
            self._smoothed_delta * FPS * INTERPOLATION_FACTOR * self.playback_speed
        )

        # Clamp to valid range. Stopping on the last frame is better than
        # spinning against the clamp: SPACE then restarts the replay.
        if self.frame_index >= self.n_frames - 1:
            self.frame_index = float(self.n_frames - 1)
            self.paused = True
        elif self.frame_index < 0.0:
            self.frame_index = 0.0

    def on_key_press(self, symbol: int, modifiers: int):
        """Handle keyboard input."""
        # While the chat box has focus every printable key belongs to the message
        # being typed, so only the editing keys are handled here. Without this
        # guard, typing "restart" would pause the replay (SPACE), jump it back to
        # the start (R) and toggle two panels (T, M) along the way.
        if self.chat_input_active and self.show_chat_panel:
            if symbol == arcade.key.ESCAPE:
                self.show_chat_panel = False
                self.chat_input_active = False
                self.chat_input = ""
            elif symbol in (arcade.key.ENTER, arcade.key.RETURN):
                if self.chat_input.strip():
                    self._send_chat_message()
            elif symbol == arcade.key.BACKSPACE:
                self.chat_input = self.chat_input[:-1]
            return

        # Seek distance in stored frames. Derived from seconds so it stays the
        # same on-screen jump no matter how INTERPOLATION_FACTOR is tuned - the
        # old fixed 10-frame step barely moved at high interpolation factors.
        seek_frames = SEEK_SECONDS * FPS * INTERPOLATION_FACTOR

        if symbol == arcade.key.SPACE:
            # Resuming after the replay ran to the end restarts it from the top.
            if self.paused and self.frame_index >= self.n_frames - 1:
                self.frame_index = 0.0
            self.paused = not self.paused
        elif symbol == arcade.key.RIGHT:
            self.frame_index = min(self.frame_index + seek_frames, self.n_frames - 1)
        elif symbol == arcade.key.LEFT:
            self.frame_index = max(self.frame_index - seek_frames, 0.0)
        elif symbol == arcade.key.R:
            # Restart from the opening frame at normal speed.
            self.frame_index = 0.0
            self.playback_speed = 1.0
            self.paused = False
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
                # Close the chat panel when it is open but not focused.
                self.show_chat_panel = False
                self.chat_input_active = False
                self.chat_input = ""

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

        # The cached trend belongs to the previous driver; clear it and let
        # _update_ml_outputs() refill it on its next tick.
        self._selected_driver_prediction = None


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
