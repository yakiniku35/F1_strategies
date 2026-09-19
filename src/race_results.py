"""
Final classification and championship points.

Turns the last frame of a race (simulated or replayed) into a finishing order
with gaps, retirement reasons and World Championship points, and renders or
exports it.

The whole module works on the plain frame dictionaries the rest of the project
already passes around, so it needs nothing from FastF1 or the ML models.
"""

import csv
import json
import os
from typing import Optional

# Points for the top ten finishers, as used by Formula 1 since 2010.
POINTS_BY_POSITION = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)

# A bonus point for the fastest lap, which only counts if the driver who set it
# also finished inside the points.
FASTEST_LAP_POINT = 1
FASTEST_LAP_MAX_POSITION = 10

# Fallback used to turn a distance gap into a time gap when the leader's speed
# is unknown or nonsensical (km/h).
DEFAULT_GAP_SPEED_KMH = 200.0

# A driver who covered less than this fraction of the winner's race distance is
# not classified, matching the real 90% rule.
CLASSIFIED_DISTANCE_FRACTION = 0.90


def _driver_lap(entry: dict) -> int:
    """Read a driver's lap number defensively (telemetry sometimes carries None)."""
    try:
        return int(entry.get("lap", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _driver_dist(entry: dict) -> float:
    """Read a driver's covered distance defensively."""
    try:
        return float(entry.get("dist", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _is_retired(entry: dict, code: str, retirements: Optional[dict]) -> bool:
    """Decide whether a driver is out of the race.

    Two independent signals are accepted: an explicit retirements mapping from
    the simulator, and the ``rel_dist == 1`` marker that the telemetry pipeline
    and the replay window already use for a car that is no longer running.
    """
    if retirements and code in retirements:
        return True
    try:
        return float(entry.get("rel_dist", 0)) == 1
    except (TypeError, ValueError):
        return False


def compute_lap_times(frames: list) -> dict:
    """Derive per-lap times for every driver from a list of frames.

    A driver's lap time is the gap between the frame where their lap counter
    ticks over and the frame where it ticked over previously. The very first
    lap is skipped, because the frames do not cover the time spent on the grid.

    Args:
        frames: Race frames, each with ``t`` and ``drivers``.

    Returns:
        Mapping of driver code -> list of ``(lap_number, lap_time_seconds)``
        for the laps that could be measured.
    """
    lap_times = {}
    last_lap = {}
    last_change_time = {}

    for frame in frames:
        frame_time = float(frame.get("t", 0.0))

        for code, entry in frame.get("drivers", {}).items():
            lap = _driver_lap(entry)
            previous = last_lap.get(code)

            if previous is None:
                last_lap[code] = lap
                last_change_time[code] = frame_time
                continue

            if lap > previous:
                started_at = last_change_time.get(code)
                if started_at is not None:
                    duration = frame_time - started_at
                    if duration > 0:
                        lap_times.setdefault(code, []).append((previous, duration))
                last_lap[code] = lap
                last_change_time[code] = frame_time

    return lap_times


def find_fastest_lap(lap_times: dict) -> Optional[dict]:
    """Find the fastest lap of the race.

    Args:
        lap_times: Output of compute_lap_times().

    Returns:
        ``{"code": ..., "lap": ..., "time": ...}`` for the quickest lap, or
        None when no lap could be measured.
    """
    best = None

    for code, laps in lap_times.items():
        for lap, duration in laps:
            if best is None or duration < best["time"]:
                best = {"code": code, "lap": lap, "time": duration}

    return best


def _format_gap(laps_down: int, seconds: Optional[float], is_leader: bool,
                retired: bool, classified: bool) -> str:
    """Render the gap column the way a results sheet does."""
    if retired:
        return "DNF"
    if not classified:
        return "NC"
    if is_leader:
        return "WINNER"
    if laps_down >= 1:
        return f"+{laps_down} LAP" if laps_down == 1 else f"+{laps_down} LAPS"
    if seconds is None:
        return "-"
    return f"+{seconds:.3f}s"


def build_classification(final_frame: dict,
                         retirements: Optional[dict] = None,
                         fastest_lap: Optional[dict] = None,
                         total_laps: Optional[int] = None) -> list:
    """Build the final classification from the last frame of a race.

    Finishers are ordered by laps completed and then by distance covered.
    Retired drivers are placed behind every finisher, ordered by how far they
    got, which is how a real results sheet reads.

    Args:
        final_frame: The last frame of the race.
        retirements: Optional ``code -> {"lap": int, "cause": str}`` from the
            simulator. Drivers marked ``rel_dist == 1`` are detected without it.
        fastest_lap: Optional result of find_fastest_lap(), for the bonus point.
        total_laps: Scheduled race distance, used for the laps column.

    Returns:
        List of result rows, already in finishing order, each with position,
        code, laps, status, cause, gap, gap_text, points and fastest_lap.
    """
    drivers = (final_frame or {}).get("drivers", {})
    if not drivers:
        return []

    retirements = retirements or {}

    finishers = []
    retired = []

    for code, entry in drivers.items():
        record = {
            "code": code,
            "laps": _driver_lap(entry),
            "dist": _driver_dist(entry),
            "speed": float(entry.get("speed", 0.0) or 0.0),
        }
        if _is_retired(entry, code, retirements):
            # Prefer the simulator's retirement lap: the frozen telemetry keeps
            # reporting the lap the car stopped on, but the cause only lives in
            # the retirements mapping.
            info = retirements.get(code, {})
            record["laps"] = int(info.get("lap", record["laps"]))
            record["cause"] = info.get("cause", "Retired")
            retired.append(record)
        else:
            record["cause"] = None
            finishers.append(record)

    finishers.sort(key=lambda r: (r["laps"], r["dist"]), reverse=True)
    retired.sort(key=lambda r: (r["laps"], r["dist"]), reverse=True)

    classification = []
    leader = finishers[0] if finishers else None
    leader_speed = leader["speed"] if leader and leader["speed"] > 1 else DEFAULT_GAP_SPEED_KMH
    # Distance per lap, inferred from the leader so this works for both the
    # simulator's synthetic units and real telemetry metres.
    lap_distance = (leader["dist"] / leader["laps"]) if leader and leader["laps"] > 0 else 0.0

    fastest_code = (fastest_lap or {}).get("code")

    for index, record in enumerate(finishers + retired):
        is_retired = record["cause"] is not None
        is_leader = (not is_retired) and index == 0

        laps_down = 0
        gap_seconds = None

        if leader and not is_retired and not is_leader:
            laps_down = max(0, leader["laps"] - record["laps"])
            distance_gap = leader["dist"] - record["dist"]
            if laps_down == 0 and distance_gap >= 0:
                # km/h -> km/s, matching the gap estimate the live leaderboard uses.
                gap_seconds = (distance_gap / 1000.0) / (leader_speed / 3600.0)

        classified = True
        if leader and not is_retired and lap_distance > 0:
            classified = record["dist"] >= leader["dist"] * CLASSIFIED_DISTANCE_FRACTION

        position = index + 1
        points = 0
        if not is_retired and classified and position <= len(POINTS_BY_POSITION):
            points = POINTS_BY_POSITION[position - 1]

        has_fastest_lap = fastest_code is not None and record["code"] == fastest_code
        if (has_fastest_lap and not is_retired
                and position <= FASTEST_LAP_MAX_POSITION):
            points += FASTEST_LAP_POINT

        classification.append({
            "position": position,
            "code": record["code"],
            "laps": record["laps"] if total_laps is None else min(record["laps"], total_laps),
            "status": "DNF" if is_retired else ("Finished" if classified else "Not classified"),
            "cause": record["cause"],
            "gap": gap_seconds,
            "laps_down": laps_down,
            "gap_text": _format_gap(laps_down, gap_seconds, is_leader, is_retired, classified),
            "points": points,
            "fastest_lap": has_fastest_lap,
        })

    return classification


def classification_from_frames(frames: list,
                               retirements: Optional[dict] = None,
                               total_laps: Optional[int] = None) -> dict:
    """Build a full result, including the fastest lap, from a whole race.

    Args:
        frames: Every frame of the race.
        retirements: Optional retirements mapping from the simulator.
        total_laps: Scheduled race distance.

    Returns:
        ``{"classification": [...], "fastest_lap": {...} or None}``.
    """
    if not frames:
        return {"classification": [], "fastest_lap": None}

    fastest = find_fastest_lap(compute_lap_times(frames))

    return {
        "classification": build_classification(
            frames[-1], retirements=retirements, fastest_lap=fastest,
            total_laps=total_laps,
        ),
        "fastest_lap": fastest,
    }


def format_classification(classification: list,
                          title: str = "FINAL CLASSIFICATION",
                          driver_names: Optional[dict] = None,
                          fastest_lap: Optional[dict] = None) -> str:
    """Render the classification as a plain-text table.

    Args:
        classification: Rows from build_classification().
        title: Heading printed above the table.
        driver_names: Optional ``code -> display name`` mapping.
        fastest_lap: Optional fastest-lap record, printed as a footer.

    Returns:
        The formatted table, ready to print.
    """
    if not classification:
        return f"{title}\n(no results available)"

    driver_names = driver_names or {}

    rows = []
    for entry in classification:
        code = entry["code"]
        name = driver_names.get(code, code)
        status = entry["gap_text"]
        if entry["cause"]:
            status = f"DNF ({entry['cause']})"

        rows.append([
            entry["position"] if entry["status"] != "DNF" else "-",
            code,
            name,
            entry["laps"],
            status,
            entry["points"] or "",
            "⏱" if entry["fastest_lap"] else "",
        ])

    headers = ["Pos", "Code", "Driver", "Laps", "Gap / Status", "Pts", "FL"]

    try:
        from tabulate import tabulate
        table = tabulate(rows, headers=headers, tablefmt="grid")
    except Exception:
        # tabulate is optional; fall back to fixed-width columns.
        widths = [max(len(str(row[i])) for row in [headers] + rows)
                  for i in range(len(headers))]
        lines = ["  ".join(str(h).ljust(w) for h, w in zip(headers, widths)),
                 "  ".join("-" * w for w in widths)]
        lines += ["  ".join(str(c).ljust(w) for c, w in zip(row, widths)) for row in rows]
        table = "\n".join(lines)

    output = [f"\n🏁 {title}", "=" * 60, table]

    if fastest_lap:
        name = driver_names.get(fastest_lap["code"], fastest_lap["code"])
        output.append(f"\n⏱  Fastest lap: {name} "
                      f"- lap {fastest_lap['lap']} ({fastest_lap['time']:.3f}s)")

    return "\n".join(output)


def export_classification(classification: list, filepath: str,
                          fastest_lap: Optional[dict] = None) -> bool:
    """Write the classification to a .json or .csv file.

    Args:
        classification: Rows from build_classification().
        filepath: Destination path; the extension selects the format.
        fastest_lap: Optional fastest-lap record (JSON output only).

    Returns:
        True on success, False if the file could not be written.
    """
    if not classification:
        return False

    directory = os.path.dirname(os.path.abspath(filepath))
    try:
        os.makedirs(directory, exist_ok=True)

        if filepath.lower().endswith(".csv"):
            columns = ["position", "code", "laps", "status", "cause",
                       "gap_text", "points", "fastest_lap"]
            with open(filepath, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns,
                                        extrasaction="ignore")
                writer.writeheader()
                writer.writerows(classification)
        else:
            with open(filepath, "w", encoding="utf-8") as handle:
                json.dump({"classification": classification,
                           "fastest_lap": fastest_lap},
                          handle, indent=2, ensure_ascii=False)

        return True

    except OSError as exc:
        print(f"⚠️ 無法寫入 {filepath}: {exc}")
        return False
