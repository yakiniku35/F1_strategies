"""Historical race replay window (the f1-race-replay derived viewer).

The telemetry behind this window is stored at FPS (25) frames per second while
the window renders at 60, so every drawn frame interpolates between two stored
ones - otherwise the cars step forward 25 times a second and the motion reads as
stutter no matter how fast the machine is.
"""

import os
import arcade
import numpy as np
from src.external_f1_data import FPS

# Kept these as "default" starting sizes, but they are no longer hard limits
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1200
SCREEN_TITLE = "F1 Replay"

# --- Animation / frame pacing -------------------------------------------------
# Arcade schedules on_update() and on_draw() at this rate. Pinning both keeps the
# simulation step and the render step on the same cadence.
TARGET_FPS = 60

# After a stall (garbage collection, a window drag) arcade reports the whole gap
# as a single delta_time. Advancing playback by it teleports the cars, so cap it.
MAX_FRAME_DELTA = 1.0 / 15.0  # seconds

# Frame times wobble by a millisecond or two even on a healthy display, which
# shows up as shimmer in the car positions. Advance by an exponential moving
# average instead: it converges on the real frame time, so playback speed is
# unchanged and only the jitter is removed.
DELTA_SMOOTHING = 0.12  # 0.0 = never adapt, 1.0 = no smoothing at all

# How much race time the left/right arrow keys skip.
SEEK_SECONDS = 3.0

# Car markers: a team-coloured body inside a white ring.
CAR_RADIUS = 6
CAR_OUTLINE_WIDTH = 2

# Track status code -> outline colour. Defined at module level so it is built
# once at import instead of being rebuilt inside every on_draw() call.
TRACK_STATUS_COLORS = {
    "1": (150, 150, 150),  # green flag / normal
    "2": (220, 180, 0),    # yellow flag
    "4": (180, 100, 30),   # safety car
    "5": (200, 30, 30),    # red flag
    "6": (200, 130, 50),   # virtual safety car
    "7": (200, 130, 50),   # VSC ending
}
DEFAULT_TRACK_COLOR = (150, 150, 150)

# Telemetry keys that may be blended between two stored frames. Everything else
# (lap, tyre, gear, DRS, classified position, the retirement marker) is a
# discrete state where blending would produce meaningless values.
INTERPOLATED_KEYS = ("x", "y", "dist", "speed")

def build_track_from_example_lap(example_lap, track_width=200):
    plot_x_ref = example_lap["X"].to_numpy()
    plot_y_ref = example_lap["Y"].to_numpy()

    # compute tangents
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

    # world bounds
    x_min = min(plot_x_ref.min(), x_inner.min(), x_outer.min())
    x_max = max(plot_x_ref.max(), x_inner.max(), x_outer.max())
    y_min = min(plot_y_ref.min(), y_inner.min(), y_outer.min())
    y_max = max(plot_y_ref.max(), y_inner.max(), y_outer.max())

    return (plot_x_ref, plot_y_ref, x_inner, y_inner, x_outer, y_outer,
            x_min, x_max, y_min, y_max)


class F1ReplayWindow(arcade.Window):
    def __init__(self, frames, track_statuses, example_lap, drivers, title,
                 playback_speed=1.0, driver_colors=None, circuit_rotation=0.0,
                 left_ui_margin=340, right_ui_margin=260, total_laps=None):
        # Resizable so the user can adjust mid-sim. vsync removes tearing and
        # pinning both rates keeps update and draw on one even cadence.
        super().__init__(
            SCREEN_WIDTH, SCREEN_HEIGHT, title,
            resizable=True,
            vsync=True,
            update_rate=1 / TARGET_FPS,
            draw_rate=1 / TARGET_FPS,
        )

        self.frames = frames
        self.track_statuses = track_statuses
        self.n_frames = len(frames)
        self.drivers = list(drivers)
        self.playback_speed = playback_speed
        self.driver_colors = driver_colors or {}
        self.frame_index = 0.0  # use float for fractional-frame accumulation
        self.paused = False
        self._tyre_textures = {}
        self.total_laps = total_laps
        self.has_weather = any("weather" in frame for frame in frames) if frames else False

        # Rotation (degrees) to apply to the whole circuit around its centre
        self.circuit_rotation = circuit_rotation
        self._rot_rad = float(np.deg2rad(self.circuit_rotation)) if self.circuit_rotation else 0.0
        self._cos_rot = float(np.cos(self._rot_rad))
        self._sin_rot = float(np.sin(self._rot_rad))
        self.finished_drivers = []
        self.left_ui_margin = left_ui_margin
        self.right_ui_margin = right_ui_margin

        # --- Animation state -------------------------------------------------
        # Low-pass filtered frame delta used to advance frame_index (see on_update).
        self._smoothed_delta = 1.0 / TARGET_FPS
        # Memoised interpolated frame, so all the panels in one rendered frame
        # share a single computation and therefore one set of positions.
        self._frame_state_cache_key = None
        self._frame_state_cache = None

        # --- Render caches ---------------------------------------------------
        # Long-lived arcade.Text objects keyed by call site (see _get_text()).
        self._text_cache = {}
        self._text_colors = {}
        # Track outline batched into a ShapeElementList, rebuilt only when the
        # flag colour or the window size changes.
        self._track_shapes = None
        self._last_track_color = None
        # Car markers as two batched SpriteLists (rings, then bodies).
        self._car_sprites = None
        self._car_outline_sprites = None
        self._car_sprite_map = {}
        # Along-track progress per driver, recomputed once per stored frame.
        self._progress_cache_idx = None
        self._progress_cache = {}

        # Import the tyre textures from the images/tyres folder (all files)
        tyres_folder = os.path.join("images", "tyres")
        if os.path.exists(tyres_folder):
            for filename in os.listdir(tyres_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    texture_name = os.path.splitext(filename)[0]
                    texture_path = os.path.join(tyres_folder, filename)
                    self._tyre_textures[texture_name] = arcade.load_texture(texture_path)

        # Build track geometry (Raw World Coordinates)
        (self.plot_x_ref, self.plot_y_ref,
         self.x_inner, self.y_inner,
         self.x_outer, self.y_outer,
         self.x_min, self.x_max,
         self.y_min, self.y_max) = build_track_from_example_lap(example_lap)

        # Rotation centre of the circuit. Cached here because world_to_screen()
        # used to recompute it on every single call - including once per point
        # for the 4000 polyline points rebuilt on every resize.
        self._world_cx = (self.x_min + self.x_max) / 2
        self._world_cy = (self.y_min + self.y_max) / 2

        # Build a dense reference polyline (used for projecting car (x,y) -> along-track distance)
        ref_points = self._interpolate_points(self.plot_x_ref, self.plot_y_ref, interp_points=4000)
        # store as numpy arrays for vectorized ops
        self._ref_xs = np.array([p[0] for p in ref_points])
        self._ref_ys = np.array([p[1] for p in ref_points])

        # cumulative distances along the reference polyline (metres)
        diffs = np.sqrt(np.diff(self._ref_xs)**2 + np.diff(self._ref_ys)**2)
        self._ref_seg_len = diffs
        self._ref_cumdist = np.concatenate(([0.0], np.cumsum(diffs)))
        self._ref_total_length = float(self._ref_cumdist[-1]) if len(self._ref_cumdist) > 0 else 0.0

        # Pre-calculate interpolated world points ONCE (optimization)
        self.world_inner_points = self._interpolate_points(self.x_inner, self.y_inner)
        self.world_outer_points = self._interpolate_points(self.x_outer, self.y_outer)

        # These will hold the actual screen coordinates to draw
        self.screen_inner_points = []
        self.screen_outer_points = []
        
        # Scaling parameters (initialized to 0, calculated in update_scaling)
        self.world_scale = 1.0
        self.tx = 0
        self.ty = 0

        # Load Background
        bg_path = os.path.join("resources", "background.png")
        self.bg_texture = arcade.load_texture(bg_path) if os.path.exists(bg_path) else None

        arcade.set_background_color(arcade.color.BLACK)

        # Trigger initial scaling calculation
        self.update_scaling(self.width, self.height)

        # Selection & hit-testing state for leaderboard
        self.selected_driver = None
        self.leaderboard_rects = []  # list of tuples: (code, left, bottom, right, top)

        # Car sprites need a live GL context, so build them last.
        self._init_car_sprites()

    def _init_car_sprites(self):
        """Create the sprites used to draw the cars.

        Each car is a white ring sprite with a team-coloured body sprite on top,
        so the whole field costs two batched SpriteList draw calls per frame
        instead of one immediate-mode draw_circle_filled() call per car.
        """
        self._car_outline_sprites = arcade.SpriteList()
        self._car_sprites = arcade.SpriteList()
        self._car_sprite_map = {}

        for code in self.drivers:
            color = self.driver_colors.get(code, arcade.color.WHITE)
            outline = arcade.SpriteCircle(CAR_RADIUS + CAR_OUTLINE_WIDTH, arcade.color.WHITE)
            body = arcade.SpriteCircle(CAR_RADIUS, color)
            # Hidden until _update_car_sprites() places them.
            outline.visible = False
            body.visible = False
            self._car_outline_sprites.append(outline)
            self._car_sprites.append(body)
            self._car_sprite_map[code] = (outline, body)

    def _get_text(self, key, text, x, y, color, font_size=12, bold=False,
                  anchor_x="left", anchor_y="baseline"):
        """Return a reusable arcade.Text object for one call site.

        Constructing an arcade.Text lays the string out glyph by glyph and
        uploads it to the GPU. Rebuilding every label on every frame was the
        single most expensive thing this window did. Each call site now keeps one
        long-lived object and only the attributes that changed are written back.

        Args:
            key: Unique, stable identifier for the call site (e.g. "hud.lap").
            text: The string to display.
            x, y: Screen position.
            color: Text colour as an RGB or RGBA tuple.
            font_size, bold, anchor_x, anchor_y: Applied once, at construction -
                these cannot change for a given key.

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

        # Each assignment invalidates arcade's cached layout even when the value
        # is unchanged, so guard every one of them.
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

    def _get_frame_state(self, frame_index):
        """Build the race state for a *fractional* frame index.

        Telemetry is stored at FPS (25) frames per second but the window renders
        at 60, so most rendered frames fall between two stored ones. Continuous
        values (track position, distance, speed, the race clock) are blended
        towards the next stored frame so the cars glide; discrete values (lap,
        tyre, gear, DRS, classified position, the retirement marker) are taken
        from the current frame unchanged, because a blended "lap 12.4" is
        meaningless.

        The result is memoised per index so the HUD, leaderboard and car pass
        share one computation and one set of positions.

        Args:
            frame_index: Fractional index into self.frames.

        Returns:
            A frame dict in the same shape as the stored frames.
        """
        if self._frame_state_cache_key == frame_index:
            return self._frame_state_cache

        idx = max(0, min(int(frame_index), self.n_frames - 1))
        next_idx = min(idx + 1, self.n_frames - 1)
        current = self.frames[idx]
        # Nothing to blend into on the very last frame of the replay.
        blend = (frame_index - idx) if next_idx != idx else 0.0

        if blend <= 0.0:
            state = current
        else:
            next_frame = self.frames[next_idx]
            drivers = {}

            for code, pos in current["drivers"].items():
                next_pos = next_frame["drivers"].get(code)
                if next_pos is None:
                    # Driver absent from the next frame - nothing to blend to.
                    drivers[code] = pos
                    continue

                blended = dict(pos)
                for key in INTERPOLATED_KEYS:
                    start = pos.get(key, 0.0)
                    blended[key] = start + (next_pos.get(key, start) - start) * blend
                drivers[code] = blended

            state = {
                "t": current["t"] + (next_frame["t"] - current["t"]) * blend,
                "lap": current.get("lap", 1),
                "drivers": drivers,
            }
            # Weather is a discrete snapshot; carry the current one across.
            if "weather" in current:
                state["weather"] = current["weather"]

        self._frame_state_cache_key = frame_index
        self._frame_state_cache = state
        return state

    def _build_track_shapes(self, track_color):
        """Batch the track outline into a single ShapeElementList.

        The outline is two 2000-point line strips. Drawing them with
        arcade.draw_line_strip() re-tessellated and re-uploaded 4000 points on
        every frame; a ShapeElementList uploads once and redraws from the GPU
        buffer. It only has to be rebuilt when the flag colour changes or the
        window is resized.

        Args:
            track_color: RGB colour for the current track status.
        """
        self._track_shapes = arcade.shape_list.ShapeElementList()

        if len(self.screen_inner_points) > 1:
            self._track_shapes.append(arcade.shape_list.create_line_strip(
                self.screen_inner_points, track_color, 4))
        if len(self.screen_outer_points) > 1:
            self._track_shapes.append(arcade.shape_list.create_line_strip(
                self.screen_outer_points, track_color, 4))

        self._last_track_color = track_color

    def _update_car_sprites(self, frame):
        """Move every car sprite to its screen position for this frame.

        Args:
            frame: Interpolated race state from _get_frame_state().

        Returns:
            Mapping of driver code -> (screen_x, screen_y) for the visible cars.
        """
        screen_positions = {}

        for code, pos in frame["drivers"].items():
            sprites = self._car_sprite_map.get(code)
            if sprites is None:
                continue

            outline, body = sprites
            sx, sy = self.world_to_screen(pos.get("x", 0.0), pos.get("y", 0.0))
            outline.position = (sx, sy)
            body.position = (sx, sy)
            outline.visible = True
            body.visible = True
            screen_positions[code] = (sx, sy)

        return screen_positions

    def _project_to_reference(self, x, y):
        """Project a world position onto the dense reference polyline.

        Returns the distance in metres travelled along the racing line to reach
        the point on it nearest to (x, y).
        """
        if self._ref_total_length == 0.0 or len(self._ref_xs) == 0:
            return 0.0

        # Nearest dense sample. Kept as a per-driver scan on purpose: batching
        # all drivers into one (n_drivers, 4000) distance matrix was measurably
        # *slower*, because the temporaries no longer fit in cache.
        dx = self._ref_xs - x
        dy = self._ref_ys - y
        idx = int(np.argmin(dx * dx + dy * dy))

        # Refine onto the adjacent segment so the answer does not quantise to
        # the polyline's sampling interval.
        if idx < len(self._ref_xs) - 1:
            x1, y1 = self._ref_xs[idx], self._ref_ys[idx]
            vx = self._ref_xs[idx + 1] - x1
            vy = self._ref_ys[idx + 1] - y1
            seg_len2 = vx * vx + vy * vy
            if seg_len2 > 0:
                t = ((x - x1) * vx + (y - y1) * vy) / seg_len2
                t_clamped = max(0.0, min(1.0, t))
                # Distance along the segment is just the clamped parameter
                # multiplied by the segment length.
                return float(self._ref_cumdist[idx] + t_clamped * np.sqrt(seg_len2))

        # Fallback: the cumulative distance at the closest dense sample.
        return float(self._ref_cumdist[idx])

    def _driver_progress(self, frame_idx, frame):
        """Along-track progress in metres for every driver.

        Ordering the leaderboard by the raw "dist" telemetry field disagrees with
        where the cars are actually drawn, so each car's (x, y) is projected onto
        a dense reference polyline of the racing line instead.

        That projection is the most expensive non-drawing work in this window:
        one scan of a 4000-point polyline per driver. It used to run on every
        *rendered* frame - sixty times a second. Telemetry only changes FPS (25)
        times a second, so the whole result is now cached against the stored
        frame index and the frames rendered in between reuse it.

        Args:
            frame_idx: Index of the stored frame, used as the cache key.
            frame: The stored frame whose driver coordinates are projected.

        Returns:
            Mapping of driver code -> metres of race distance covered.
        """
        if self._progress_cache_idx == frame_idx:
            return self._progress_cache

        progress = {}

        for code, pos in frame["drivers"].items():
            # Parse the lap defensively - telemetry occasionally carries None.
            try:
                lap = int(pos.get("lap", 1))
            except (TypeError, ValueError):
                lap = 1

            projected_m = self._project_to_reference(pos.get("x", 0.0), pos.get("y", 0.0))
            # Distance since the start = completed laps + progress around this lap.
            progress[code] = float((max(lap, 1) - 1) * self._ref_total_length + projected_m)

        self._progress_cache_idx = frame_idx
        self._progress_cache = progress
        return progress

    def _interpolate_points(self, xs, ys, interp_points=2000):
        t_old = np.linspace(0, 1, len(xs))
        t_new = np.linspace(0, 1, interp_points)
        xs_i = np.interp(t_new, t_old, xs)
        ys_i = np.interp(t_new, t_old, ys)
        return list(zip(xs_i, ys_i))

    def update_scaling(self, screen_w, screen_h):
        """
        Recalculates the scale and translation to fit the track 
        perfectly within the new screen dimensions while maintaining aspect ratio.
        """
        padding = 0.05
        # If a rotation is applied, we must compute the rotated bounds
        world_cx = self._world_cx
        world_cy = self._world_cy

        def _rotate_about_center(x, y):
            # Translate to centre, rotate, translate back
            tx = x - world_cx
            ty = y - world_cy
            rx = tx * self._cos_rot - ty * self._sin_rot
            ry = tx * self._sin_rot + ty * self._cos_rot
            return rx + world_cx, ry + world_cy

        # Build rotated extents from inner/outer world points
        rotated_points = []
        for x, y in self.world_inner_points:
            rotated_points.append(_rotate_about_center(x, y))
        for x, y in self.world_outer_points:
            rotated_points.append(_rotate_about_center(x, y))

        xs = [p[0] for p in rotated_points]
        ys = [p[1] for p in rotated_points]
        world_x_min = min(xs) if xs else self.x_min
        world_x_max = max(xs) if xs else self.x_max
        world_y_min = min(ys) if ys else self.y_min
        world_y_max = max(ys) if ys else self.y_max

        world_w = max(1.0, world_x_max - world_x_min)
        world_h = max(1.0, world_y_max - world_y_min)
        
        # Reserve left/right UI margins before applying padding so the track
        # never overlaps side UI elements (leaderboard, telemetry, legends).
        inner_w = max(1.0, screen_w - self.left_ui_margin - self.right_ui_margin)
        usable_w = inner_w * (1 - 2 * padding)
        usable_h = screen_h * (1 - 2 * padding)

        # Calculate scale to fit whichever dimension is the limiting factor
        scale_x = usable_w / world_w
        scale_y = usable_h / world_h
        self.world_scale = min(scale_x, scale_y)

        # Center the world in the screen (rotation done about original centre)
        # world_cx/world_cy are unchanged by rotation about centre
        # Center within the available inner area (left_ui_margin .. screen_w - right_ui_margin)
        screen_cx = self.left_ui_margin + inner_w / 2
        screen_cy = screen_h / 2

        self.tx = screen_cx - self.world_scale * world_cx
        self.ty = screen_cy - self.world_scale * world_cy

        # Update the polyline screen coordinates based on new scale
        self.screen_inner_points = [self.world_to_screen(x, y) for x, y in self.world_inner_points]
        self.screen_outer_points = [self.world_to_screen(x, y) for x, y in self.world_outer_points]

        # The batched track geometry holds screen coordinates, so it is now stale.
        self._track_shapes = None
        self._last_track_color = None

    def on_resize(self, width, height):
        """Called automatically by Arcade when window is resized."""
        super().on_resize(width, height)
        self.update_scaling(width, height)

    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates.

        Rotates around the cached track centre (when a circuit rotation is set),
        then applies the fit scale and offset computed by update_scaling().
        """
        if self._rot_rad:
            tx = x - self._world_cx
            ty = y - self._world_cy
            rx = tx * self._cos_rot - ty * self._sin_rot
            ry = tx * self._sin_rot + ty * self._cos_rot
            x, y = rx + self._world_cx, ry + self._world_cy

        sx = self.world_scale * x + self.tx
        sy = self.world_scale * y + self.ty
        return sx, sy

    def _format_wind_direction(self, degrees):
        if degrees is None:
            return "N/A"
        deg_norm = degrees % 360
        dirs = [
            "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
        ]
        idx = int((deg_norm / 22.5) + 0.5) % len(dirs)
        return dirs[idx]

    def on_draw(self):
        self.clear()

        # 1. Draw Background (stretched to fit new window size)
        if self.bg_texture:
            arcade.draw_texture_rect(
                texture=self.bg_texture,
                rect=arcade.LBWH(0, 0, self.width, self.height),
            )

        # 2. Build the interpolated race state once and share it with every panel
        # below, so the cars, the HUD and the leaderboard describe one instant.
        idx = min(int(self.frame_index), self.n_frames - 1)
        frame = self._get_frame_state(self.frame_index)
        current_time = frame["t"]

        current_track_status = "1"  # default: green flag
        for status in self.track_statuses:
            if status['start_time'] <= current_time and (status['end_time'] is None or current_time < status['end_time']):
                current_track_status = status['status']
                break

        track_color = TRACK_STATUS_COLORS.get(current_track_status, DEFAULT_TRACK_COLOR)

        # 3. Draw Track from the batched shape list, rebuilding it only when the
        # flag colour changes (a resize clears the cache in update_scaling()).
        if self._track_shapes is None or self._last_track_color != track_color:
            self._build_track_shapes(track_color)
        if self._track_shapes:
            self._track_shapes.draw()

        # 4. Draw Cars - two batched SpriteList calls (rings, then bodies).
        self._update_car_sprites(frame)
        self._car_outline_sprites.draw()
        self._car_sprites.draw()

        # --- UI ELEMENTS (Dynamic Positioning) ---

        # Order the leaderboard by projected along-track distance rather than the
        # raw "dist" field, so it agrees with what is drawn on track. Computed
        # once per stored frame and cached (see _driver_progress).
        driver_progress = self._driver_progress(idx, self.frames[idx])

        # Leader is the one with greatest progress_m
        if driver_progress:
            leader_code = max(driver_progress, key=lambda c: driver_progress[c])
            leader_lap = frame["drivers"][leader_code].get("lap", 1)
        else:
            leader_code = None
            leader_lap = 1

        # Time Calculation
        t = frame["t"]
        hours = int(t // 3600)
        minutes = int((t % 3600) // 60)
        seconds = int(t % 60)
        time_str = f"{hours:02}:{minutes:02}:{seconds:02}"

        # Format Lap String 
        lap_str = f"Lap: {leader_lap}"
        if self.total_laps is not None:
            lap_str += f"/{self.total_laps}"

        # Draw HUD - Top Left
        self._get_text("hud.lap", lap_str,
                       20, self.height - 40,
                       arcade.color.WHITE, 24, anchor_y="top").draw()

        paused_suffix = "  [PAUSED]" if self.paused else ""
        self._get_text("hud.time",
                       f"Race Time: {time_str} (x{self.playback_speed}){paused_suffix}",
                       20, self.height - 80,
                       arcade.color.WHITE, 20, anchor_y="top").draw()

        # One cached label for the flag banner; only its text and colour change.
        flag_banner = {
            "2": ("YELLOW FLAG", arcade.color.YELLOW),
            "4": ("SAFETY CAR", arcade.color.BROWN),
            "5": ("RED FLAG", arcade.color.RED),
            "6": ("VIRTUAL SAFETY CAR", arcade.color.ORANGE),
            "7": ("VSC ENDING", arcade.color.ORANGE),
        }.get(current_track_status)

        if flag_banner:
            status_text, status_color = flag_banner
            self._get_text("hud.flag", status_text,
                           20, self.height - 120,
                           status_color, 24, bold=True, anchor_y="top").draw()

        # Weather Panel - Top Left block under session info
        weather_info = frame.get("weather") if frame else None
        panel_left = 20
        panel_width = 280
        panel_height = 130
        panel_top = self.height - 170
        weather_bottom = None
        if weather_info or self.has_weather:
            self._get_text(
                "weather.header",
                "Weather",
                panel_left + 12,
                panel_top - 10,
                arcade.color.WHITE,
                18,
                bold=True,
                anchor_y="top",
            ).draw()

            def _fmt(val, suffix="", precision=1):
                return f"{val:.{precision}f}{suffix}" if val is not None else "N/A"

            info = weather_info or {}
            track_temp_text = f"🌡️ Track: {_fmt(info.get('track_temp'), '°C')}"
            air_temp_text = f"🌡️ Air: {_fmt(info.get('air_temp'), '°C')}"
            humidity_text = f"💧 Humidity: {_fmt(info.get('humidity'), '%', precision=0)}"
            wind_dir_text = self._format_wind_direction(info.get('wind_direction'))
            wind_speed_text = _fmt(info.get('wind_speed'), ' km/h')
            wind_text = f" 🌬️ Wind: {wind_speed_text} {wind_dir_text}"
            rain_state = info.get('rain_state', 'N/A')
            rain_text = f"🌧️ Rain: {rain_state}"

            weather_lines = [
                track_temp_text,
                air_temp_text,
                humidity_text,
                wind_text,
                rain_text,
            ]

            start_y = panel_top - 36
            line_spacing = 22
            for line_idx, line in enumerate(weather_lines):
                self._get_text(
                    f"weather.line.{line_idx}",
                    line,
                    panel_left + 12,
                    start_y - line_idx * line_spacing,
                    arcade.color.LIGHT_GRAY,
                    14,
                    anchor_y="top",
                ).draw()
            weather_bottom = panel_top - panel_height

        # Draw Leaderboard - Top Right (inside the reserved right UI margin)
        leaderboard_x = max(20, self.width - self.right_ui_margin + 12)
        leaderboard_y = self.height - 40
        
        self._get_text("lb.header", "Leaderboard", leaderboard_x, leaderboard_y,
                       arcade.color.WHITE, 20, bold=True,
                       anchor_x="left", anchor_y="top").draw()

        driver_list = []
        for code, pos in frame["drivers"].items():
            color = self.driver_colors.get(code, arcade.color.WHITE)
            progress_m = driver_progress.get(code, float(pos.get("dist", 0.0)))
            driver_list.append((code, color, pos, progress_m))

        # Sort by computed progress (metres) so ordering matches on-track x/y positions
        driver_list.sort(key=lambda x: x[3], reverse=True)

        # Reset recorded rects each frame
        self.leaderboard_rects = []

        row_height = 25
        entry_width = 240  # clickable width for each entry
        for i, (code, color, pos, progress_m) in enumerate(driver_list):
            current_pos = i + 1
            if pos.get("rel_dist", 0) == 1:
                text = f"{current_pos}. {code}   OUT"
            else:
                text = f"{current_pos}. {code}"
    
            # Compute bounding box for this entry (match how text is positioned)
            top_y = leaderboard_y - 30 - ((current_pos - 1) * row_height)
            bottom_y = top_y - row_height
            left_x = leaderboard_x
            right_x = leaderboard_x + entry_width

            # Save for mouse hit-testing
            self.leaderboard_rects.append((code, left_x, bottom_y, right_x, top_y))

            # Highlight if selected
            if code == self.selected_driver:
                # subtle highlight behind the text
                rect = arcade.XYWH((left_x + right_x) / 2,
                    (top_y + bottom_y) / 2,
                    right_x - left_x,
                    top_y - bottom_y,)
                arcade.draw_rect_filled(
                    rect,
                    arcade.color.LIGHT_GRAY,
                )
                text_color = arcade.color.BLACK
            else:
                text_color = color

            # Cache key is the row index, not the driver: a row keeps its place
            # on screen while the driver occupying it changes.
            self._get_text(
                f"lb.row.{i}",
                text,
                left_x,
                top_y,
                text_color,
                16,
                anchor_x="left", anchor_y="top",
            ).draw()

            # Tyre Icons
            tyre_texture = self._tyre_textures.get(str(pos.get("tyre", "?")).upper())
            if tyre_texture:
                # position tyre icon inside the leaderboard area so it doesn't collide with track
                tyre_icon_x = leaderboard_x + entry_width - 10
                tyre_icon_y = top_y - 12
                icon_size = 16

                rect = arcade.XYWH(tyre_icon_x, tyre_icon_y, icon_size, icon_size)

                # Draw the textured rect
                arcade.draw_texture_rect(
                    rect=rect,
                    texture=tyre_texture,
                    angle=0,
                    alpha=255
                )

        # Controls Legend - Bottom Left (keeps small offset from left UI edge)
        legend_x = max(12, self.left_ui_margin - 320) if hasattr(self, "left_ui_margin") else 20
        legend_y = 150 # Height of legend block
        legend_lines = [
            "Controls:",
            "[SPACE]  Pause/Resume",
            "[←/→]    Rewind / FastForward",
            "[↑/↓]    Speed +/- (0.5x, 1x, 2x, 4x)",
            "[R]       Restart",
        ]

        
        for i, line in enumerate(legend_lines):
            self._get_text(
                f"legend.{i}",
                line,
                legend_x,
                legend_y - (i * 25),
                arcade.color.LIGHT_GRAY if i > 0 else arcade.color.WHITE,
                14,
                bold=(i == 0),
            ).draw()
        
        # Selected Driver Info - Middle Left

        if self.selected_driver and self.selected_driver in frame["drivers"]:
            # Draw box, with the driver's name in another box at the top of the original box
            driver_pos = frame["drivers"][self.selected_driver]

            driver_color = self.driver_colors.get(self.selected_driver, arcade.color.GRAY)

            info_x = 20
            default_info_y = self.height / 2 + 100
            box_width = 300
            box_height = 150
            # Keep the driver box below the weather panel if present, but above the controls legend
            if weather_bottom is not None:
                target_top = weather_bottom - 20
                info_y = min(default_info_y, target_top - box_height / 2)
            else:
                info_y = default_info_y
            min_info_y = 220  # stay above controls legend
            info_y = max(info_y, min_info_y + box_height / 2)
            
            # Background box

            bg_rect = arcade.XYWH(
                info_x + box_width / 2,
                info_y - box_height / 2,
                box_width,
                box_height
            )

            arcade.draw_rect_outline(
                bg_rect,
                driver_color
            )

            # Driver Name box
            name_rect = arcade.XYWH(
                info_x + box_width / 2,
                info_y + 20,
                box_width,
                40
            )
            arcade.draw_rect_filled(
                name_rect,
                driver_color
            )
            self._get_text(
                "driver.name",
                f"Driver: {self.selected_driver}",
                info_x + 10,
                info_y + 20,
                arcade.color.BLACK,
                16,
                anchor_x="left", anchor_y="center",
            ).draw()

            # Driver Stats from Telemetry
            speed_text = f"Speed: {driver_pos.get('speed', 0):.1f} km/h"
            gear_text = f"Gear: {driver_pos.get('gear', 0)}"
            drs_status = "off"
            drs_value = driver_pos.get('drs', 0)
            if drs_value in [0, 1]:
                drs_status = "Off"
            elif drs_value == 8:
                drs_status = "Eligible"
            elif drs_value in [10, 12, 14]:
                drs_status = "On"
            else:
                drs_status = "Unknown"
            
            drs_active_text = f"DRS: {drs_status}"
            current_lap = driver_pos.get("lap", 1)

            lap_time_text = f"Current Lap: {current_lap}"
            stats_lines = [speed_text, gear_text, drs_active_text, lap_time_text]
            for i, line in enumerate(stats_lines):
                self._get_text(
                    f"driver.stat.{i}",
                    line,
                    info_x + 10,
                    info_y - 20 - (i * 25),
                    arcade.color.WHITE,
                    14,
                    anchor_x="left", anchor_y="center",
                ).draw()
                    
    def on_update(self, delta_time: float):
        """Advance the replay clock.

        Args:
            delta_time: Real seconds since the previous update, as reported by
                arcade. It is neither capped nor evenly spaced, so both are dealt
                with here before the value is used.
        """
        # Cap outliers first: after a stall arcade reports the whole gap as one
        # delta, and advancing by it would teleport the cars across the track.
        delta_time = min(delta_time, MAX_FRAME_DELTA)

        # Then low-pass filter what is left so millisecond-scale wobble in the
        # frame times does not become visible shimmer. The average converges on
        # the true frame time, so playback speed is unaffected.
        self._smoothed_delta += (delta_time - self._smoothed_delta) * DELTA_SMOOTHING

        if self.paused:
            return

        self.frame_index += self._smoothed_delta * FPS * self.playback_speed

        # Stop cleanly on the last frame rather than spinning against the clamp;
        # SPACE then restarts the replay from the beginning.
        if self.frame_index >= self.n_frames - 1:
            self.frame_index = float(self.n_frames - 1)
            self.paused = True
        elif self.frame_index < 0.0:
            self.frame_index = 0.0

    def on_key_press(self, symbol: int, modifiers: int):
        # Seek distance in stored frames, derived from seconds so the on-screen
        # jump stays the same regardless of the telemetry frame rate. The old
        # fixed 10-frame step was less than half a second of race time.
        seek_frames = SEEK_SECONDS * FPS

        if symbol == arcade.key.SPACE:
            # Resuming after the replay ran to the end restarts it from the top.
            if self.paused and self.frame_index >= self.n_frames - 1:
                self.frame_index = 0.0
            self.paused = not self.paused
        elif symbol == arcade.key.RIGHT:
            self.frame_index = min(self.frame_index + seek_frames, self.n_frames - 1)
        elif symbol == arcade.key.LEFT:
            self.frame_index = max(self.frame_index - seek_frames, 0.0)
        elif symbol == arcade.key.UP:
            self.playback_speed = min(self.playback_speed * 2.0, 8.0)
        elif symbol == arcade.key.DOWN:
            self.playback_speed = max(0.1, self.playback_speed / 2.0)
        elif symbol == arcade.key.KEY_1:
            self.playback_speed = 0.5
        elif symbol == arcade.key.KEY_2:
            self.playback_speed = 1.0
        elif symbol == arcade.key.KEY_3:
            self.playback_speed = 2.0
        elif symbol == arcade.key.KEY_4:
            self.playback_speed = 4.0
        elif symbol == arcade.key.R:
            self.frame_index = 0.0
            self.playback_speed = 1.0
            self.paused = False

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int):
        # Default: clear selection
        new_selection = None
        for code, left, bottom, right, top in self.leaderboard_rects:
            if left <= x <= right and bottom <= y <= top:
                new_selection = code
                break

        # Toggle if clicking the same entry
        if new_selection == self.selected_driver:
            self.selected_driver = None
        else:
            self.selected_driver = new_selection

def run_arcade_replay(frames, track_statuses, example_lap, drivers, title,
                      playback_speed=1.0, driver_colors=None, circuit_rotation=0.0, total_laps=None):
    window = F1ReplayWindow(
        frames=frames,
        track_statuses=track_statuses,
        example_lap=example_lap,
        drivers=drivers,
        playback_speed=playback_speed,
        driver_colors=driver_colors,
        title=title,
        total_laps=total_laps,
        circuit_rotation=circuit_rotation,
    )
    arcade.run()
