"""
F1 Race Data processing module.
Based on f1-race-replay project by Tom Shaw.
"""

import os
import json
from datetime import timedelta
import fastf1
import fastf1.plotting
import numpy as np

from src.lib.tyres import get_tyre_compound_int

# Frame rate for replay
FPS = 25
DT = 1 / FPS


def enable_cache(cache_dir=".fastf1-cache"):
    """Enable FastF1 cache."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    fastf1.Cache.enable_cache(cache_dir)


def load_race_session(year, round_number, session_type='R'):
    """Load a race session from FastF1."""
    session = fastf1.get_session(year, round_number, session_type)
    session.load(telemetry=True)
    return session


def get_driver_colors(session):
    """Get driver colors from FastF1 plotting module."""
    try:
        color_mapping = fastf1.plotting.get_driver_color_mapping(session)
        rgb_colors = {}
        for driver, hex_color in color_mapping.items():
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
            rgb_colors[driver] = rgb
        return rgb_colors
    except Exception:
        return {}


def get_race_telemetry(session, refresh_data=False):
    """
    Extract race telemetry data from a session.
    Returns frames, driver colors, and track statuses.
    """
    event_name = str(session).replace(' ', '_').replace('/', '_')

    # Check if data has already been computed
    if not refresh_data:
        try:
            data_path = f"computed_data/{event_name}_race_telemetry.json"
            if os.path.exists(data_path):
                with open(data_path, "r") as f:
                    data = json.load(f)
                    print("Loaded precomputed race telemetry data.")
                    return data
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    drivers = session.drivers
    driver_codes = {
        num: session.get_driver(num)["Abbreviation"]
        for num in drivers
    }

    driver_data = {}
    global_t_min = None
    global_t_max = None

    # 1. Get all drivers telemetry data
    for driver_no in drivers:
        code = driver_codes[driver_no]
        print(f"Getting telemetry for driver: {code}")

        laps_driver = session.laps.pick_drivers(driver_no)
        if laps_driver.empty:
            continue

        t_all = []
        x_all = []
        y_all = []
        race_dist_all = []
        rel_dist_all = []
        lap_numbers = []
        tyre_compounds = []
        speed_all = []
        gear_all = []
        drs_all = []

        total_dist_so_far = 0.0

        for _, lap in laps_driver.iterlaps():
            lap_tel = lap.get_telemetry()
            lap_number = lap.LapNumber
            tyre_compound_int = get_tyre_compound_int(lap.Compound)

            if lap_tel.empty:
                continue

            t_lap = lap_tel["SessionTime"].dt.total_seconds().to_numpy()
            x_lap = lap_tel["X"].to_numpy()
            y_lap = lap_tel["Y"].to_numpy()
            d_lap = lap_tel["Distance"].to_numpy()
            rd_lap = lap_tel["RelativeDistance"].to_numpy()
            speed_lap = lap_tel["Speed"].to_numpy()
            gear_lap = lap_tel["nGear"].to_numpy()
            drs_lap = lap_tel["DRS"].to_numpy()

            # Normalize lap distance to start at 0
            d_lap = d_lap - d_lap.min()
            lap_length = d_lap.max()

            # Race distance = distance before this lap + distance within this lap
            race_d_lap = total_dist_so_far + d_lap
            total_dist_so_far += lap_length

            t_all.append(t_lap)
            x_all.append(x_lap)
            y_all.append(y_lap)
            race_dist_all.append(race_d_lap)
            rel_dist_all.append(rd_lap)
            lap_numbers.append(np.full_like(t_lap, lap_number))
            tyre_compounds.append(np.full_like(t_lap, tyre_compound_int))
            speed_all.append(speed_lap)
            gear_all.append(gear_lap)
            drs_all.append(drs_lap)

        if not t_all:
            continue

        t_all = np.concatenate(t_all)
        x_all = np.concatenate(x_all)
        y_all = np.concatenate(y_all)
        race_dist_all = np.concatenate(race_dist_all)
        rel_dist_all = np.concatenate(rel_dist_all)
        lap_numbers = np.concatenate(lap_numbers)
        tyre_compounds = np.concatenate(tyre_compounds)
        speed_all = np.concatenate(speed_all)
        gear_all = np.concatenate(gear_all)
        drs_all = np.concatenate(drs_all)

        order = np.argsort(t_all)
        t_all = t_all[order]
        x_all = x_all[order]
        y_all = y_all[order]
        race_dist_all = race_dist_all[order]
        rel_dist_all = rel_dist_all[order]
        lap_numbers = lap_numbers[order]
        tyre_compounds = tyre_compounds[order]
        speed_all = speed_all[order]
        gear_all = gear_all[order]
        drs_all = drs_all[order]

        driver_data[code] = {
            "t": t_all,
            "x": x_all,
            "y": y_all,
            "dist": race_dist_all,
            "rel_dist": rel_dist_all,
            "lap": lap_numbers,
            "tyre": tyre_compounds,
            "speed": speed_all,
            "gear": gear_all,
            "drs": drs_all,
        }

        t_min = t_all.min()
        t_max = t_all.max()
        global_t_min = t_min if global_t_min is None else min(global_t_min, t_min)
        global_t_max = t_max if global_t_max is None else max(global_t_max, t_max)

    # 2. Create a timeline (start from zero)
    timeline = np.arange(global_t_min, global_t_max, DT) - global_t_min

    # 3. Resample each driver's telemetry onto the common timeline
    resampled_data = {}

    for code, data in driver_data.items():
        t = data["t"] - global_t_min
        x = data["x"]
        y = data["y"]
        dist = data["dist"]
        rel_dist = data["rel_dist"]
        tyre = data["tyre"]
        speed = data["speed"]
        gear = data["gear"]
        drs = data["drs"]

        order = np.argsort(t)
        t_sorted = t[order]
        x_sorted = x[order]
        y_sorted = y[order]
        dist_sorted = dist[order]
        rel_dist_sorted = rel_dist[order]
        lap_sorted = data["lap"][order]
        tyre_sorted = tyre[order]
        speed_sorted = speed[order]
        gear_sorted = gear[order]
        drs_sorted = drs[order]

        x_resampled = np.interp(timeline, t_sorted, x_sorted)
        y_resampled = np.interp(timeline, t_sorted, y_sorted)
        dist_resampled = np.interp(timeline, t_sorted, dist_sorted)
        rel_dist_resampled = np.interp(timeline, t_sorted, rel_dist_sorted)
        lap_resampled = np.interp(timeline, t_sorted, lap_sorted)
        tyre_resampled = np.interp(timeline, t_sorted, tyre_sorted)
        speed_resampled = np.interp(timeline, t_sorted, speed_sorted)
        gear_resampled = np.interp(timeline, t_sorted, gear_sorted)
        drs_resampled = np.interp(timeline, t_sorted, drs_sorted)

        resampled_data[code] = {
            "t": timeline,
            "x": x_resampled,
            "y": y_resampled,
            "dist": dist_resampled,
            "rel_dist": rel_dist_resampled,
            "lap": lap_resampled,
            "tyre": tyre_resampled,
            "speed": speed_resampled,
            "gear": gear_resampled,
            "drs": drs_resampled,
        }

    # 4. Incorporate track status data
    formatted_track_statuses = []
    try:
        track_status = session.track_status
        for status in track_status.to_dict('records'):
            seconds = timedelta.total_seconds(status['Time'])
            start_time = seconds - global_t_min

            if formatted_track_statuses:
                formatted_track_statuses[-1]['end_time'] = start_time

            formatted_track_statuses.append({
                'status': status['Status'],
                'start_time': start_time,
                'end_time': None,
            })
    except (KeyError, AttributeError, TypeError):
        pass

    # 5. Build the frames
    frames = []
    for i, t in enumerate(timeline):
        snapshot = []
        for code, d in resampled_data.items():
            snapshot.append({
                "code": code,
                "dist": float(d["dist"][i]),
                "x": float(d["x"][i]),
                "y": float(d["y"][i]),
                "lap": int(round(d["lap"][i])),
                "rel_dist": float(d["rel_dist"][i]),
                "tyre": int(d["tyre"][i]),
                "speed": float(d["speed"][i]),
                "gear": int(d["gear"][i]),
                "drs": int(d["drs"][i]),
            })

        if not snapshot:
            continue

        # Sort by race distance to get positions
        snapshot.sort(key=lambda r: r["dist"], reverse=True)

        leader = snapshot[0]
        leader_lap = leader["lap"]

        frame_data = {}
        for idx, car in enumerate(snapshot):
            code = car["code"]
            position = idx + 1

            frame_data[code] = {
                "x": car["x"],
                "y": car["y"],
                "dist": car["dist"],
                "lap": car["lap"],
                "rel_dist": round(car["rel_dist"], 6),
                "tyre": car["tyre"],
                "position": position,
                "speed": car["speed"],
                "gear": car["gear"],
                "drs": car["drs"],
            }

        frames.append({
            "t": float(t),
            "lap": leader_lap,
            "drivers": frame_data,
        })

    print("Completed telemetry extraction...")

    # Save to file
    if not os.path.exists("computed_data"):
        os.makedirs("computed_data")

    result_data = {
        "frames": frames,
        "driver_colors": get_driver_colors(session),
        "track_statuses": formatted_track_statuses,
    }

    data_path = f"computed_data/{event_name}_race_telemetry.json"
    with open(data_path, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"Saved telemetry data to {data_path}")
    return result_data
