"""
F1 Race Data processing module.
Based on f1-race-replay project by Tom Shaw.

Optimized for performance with NumPy-based data structures.
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

# Field indices for the NumPy 3D driver data array
# Array structure: (n_frames, n_drivers, n_fields)
FIELD_X = 0
FIELD_Y = 1
FIELD_DIST = 2
FIELD_REL_DIST = 3
FIELD_LAP = 4
FIELD_TYRE = 5
FIELD_SPEED = 6
FIELD_GEAR = 7
FIELD_DRS = 8
FIELD_POSITION = 9
NUM_FIELDS = 10


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


def get_race_data(session, refresh_data=False):
    """
    Extract race telemetry data from a session using optimized NumPy arrays.
    
    Returns a dictionary containing:
        - driver_data_array: NumPy 3D array (n_frames, n_drivers, n_fields)
          Fields: x, y, dist, rel_dist, lap, tyre, speed, gear, drs, position
        - frame_metadata: NumPy 2D array (n_frames, 2) with [time, leader_lap]
        - track_statuses: List of track status dictionaries
        - driver_codes: List of driver codes (maps to driver index in array)
        - driver_colors: Dictionary mapping driver codes to RGB colors
    
    This optimized format enables efficient vector operations and 
    reduces CPU overhead compared to the dictionary-based approach.
    """
    event_name = str(session).replace(' ', '_').replace('/', '_')
    
    # Check if data has already been computed
    if not refresh_data:
        try:
            data_path = f"computed_data/{event_name}_race_data_numpy.npz"
            meta_path = f"computed_data/{event_name}_race_meta.json"
            if os.path.exists(data_path) and os.path.exists(meta_path):
                # Load NumPy arrays
                npz_data = np.load(data_path)
                driver_data_array = npz_data['driver_data']
                frame_metadata = npz_data['frame_metadata']
                
                # Load metadata (driver codes, colors, track statuses)
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                
                print("Loaded precomputed optimized race data.")
                return {
                    'driver_data_array': driver_data_array,
                    'frame_metadata': frame_metadata,
                    'track_statuses': meta['track_statuses'],
                    'driver_codes': meta['driver_codes'],
                    'driver_colors': meta['driver_colors'],
                }
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass
    
    drivers = session.drivers
    driver_codes_map = {
        num: session.get_driver(num)["Abbreviation"]
        for num in drivers
    }
    
    driver_data = {}
    global_t_min = None
    global_t_max = None
    
    # 1. Get all drivers telemetry data
    for driver_no in drivers:
        code = driver_codes_map[driver_no]
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
    n_frames = len(timeline)
    
    # Get sorted driver codes for consistent ordering
    driver_codes = sorted(driver_data.keys())
    n_drivers = len(driver_codes)
    
    # 3. Create the 3D NumPy array for driver data
    # Shape: (n_frames, n_drivers, n_fields)
    driver_data_array = np.zeros((n_frames, n_drivers, NUM_FIELDS), dtype=np.float32)
    
    # Resample each driver's telemetry onto the common timeline
    for driver_idx, code in enumerate(driver_codes):
        data = driver_data[code]
        t = data["t"] - global_t_min
        
        order = np.argsort(t)
        t_sorted = t[order]
        
        # Resample all fields using vectorized interpolation
        driver_data_array[:, driver_idx, FIELD_X] = np.interp(timeline, t_sorted, data["x"][order])
        driver_data_array[:, driver_idx, FIELD_Y] = np.interp(timeline, t_sorted, data["y"][order])
        driver_data_array[:, driver_idx, FIELD_DIST] = np.interp(timeline, t_sorted, data["dist"][order])
        driver_data_array[:, driver_idx, FIELD_REL_DIST] = np.interp(timeline, t_sorted, data["rel_dist"][order])
        driver_data_array[:, driver_idx, FIELD_LAP] = np.interp(timeline, t_sorted, data["lap"][order])
        driver_data_array[:, driver_idx, FIELD_TYRE] = np.interp(timeline, t_sorted, data["tyre"][order])
        driver_data_array[:, driver_idx, FIELD_SPEED] = np.interp(timeline, t_sorted, data["speed"][order])
        driver_data_array[:, driver_idx, FIELD_GEAR] = np.interp(timeline, t_sorted, data["gear"][order])
        driver_data_array[:, driver_idx, FIELD_DRS] = np.interp(timeline, t_sorted, data["drs"][order])
    
    # 4. Calculate positions for each frame using vectorized operations
    for frame_idx in range(n_frames):
        distances = driver_data_array[frame_idx, :, FIELD_DIST]
        # Sort by distance (descending), assign positions
        sorted_indices = np.argsort(-distances)  # Negative for descending order
        positions = np.empty(n_drivers, dtype=np.float32)
        positions[sorted_indices] = np.arange(1, n_drivers + 1, dtype=np.float32)
        driver_data_array[frame_idx, :, FIELD_POSITION] = positions
    
    # 5. Create frame metadata array (time, leader_lap)
    frame_metadata = np.zeros((n_frames, 2), dtype=np.float32)
    frame_metadata[:, 0] = timeline
    
    # Find leader lap for each frame
    for frame_idx in range(n_frames):
        positions = driver_data_array[frame_idx, :, FIELD_POSITION]
        leader_idx = np.argmin(positions)
        frame_metadata[frame_idx, 1] = driver_data_array[frame_idx, leader_idx, FIELD_LAP]
    
    # 6. Incorporate track status data
    formatted_track_statuses = []
    try:
        track_status = session.track_status
        for status in track_status.to_dict('records'):
            # status['Time'] is a timedelta object from pandas
            seconds = status['Time'].total_seconds()
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
    
    print("Completed optimized telemetry extraction...")
    
    # Save to files
    if not os.path.exists("computed_data"):
        os.makedirs("computed_data")
    
    # Save NumPy arrays
    data_path = f"computed_data/{event_name}_race_data_numpy.npz"
    np.savez_compressed(data_path, 
                        driver_data=driver_data_array,
                        frame_metadata=frame_metadata)
    
    # Save metadata as JSON
    meta_path = f"computed_data/{event_name}_race_meta.json"
    meta = {
        'driver_codes': driver_codes,
        'driver_colors': get_driver_colors(session),
        'track_statuses': formatted_track_statuses,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Saved optimized telemetry data to {data_path}")
    
    return {
        'driver_data_array': driver_data_array,
        'frame_metadata': frame_metadata,
        'track_statuses': formatted_track_statuses,
        'driver_codes': driver_codes,
        'driver_colors': meta['driver_colors'],
    }


def convert_numpy_to_frames(driver_data_array, frame_metadata, driver_codes):
    """
    Convert NumPy arrays back to the legacy frame format for backward compatibility.
    
    Args:
        driver_data_array: NumPy 3D array (n_frames, n_drivers, n_fields)
        frame_metadata: NumPy 2D array (n_frames, 2) with [time, leader_lap]
        driver_codes: List of driver codes
    
    Returns:
        List of frame dictionaries in the legacy format
    """
    frames = []
    n_frames = driver_data_array.shape[0]
    
    for frame_idx in range(n_frames):
        frame_data = {}
        for driver_idx, code in enumerate(driver_codes):
            frame_data[code] = {
                "x": float(driver_data_array[frame_idx, driver_idx, FIELD_X]),
                "y": float(driver_data_array[frame_idx, driver_idx, FIELD_Y]),
                "dist": float(driver_data_array[frame_idx, driver_idx, FIELD_DIST]),
                "rel_dist": round(float(driver_data_array[frame_idx, driver_idx, FIELD_REL_DIST]), 6),
                "lap": int(round(driver_data_array[frame_idx, driver_idx, FIELD_LAP])),
                "tyre": int(driver_data_array[frame_idx, driver_idx, FIELD_TYRE]),
                "position": int(driver_data_array[frame_idx, driver_idx, FIELD_POSITION]),
                "speed": float(driver_data_array[frame_idx, driver_idx, FIELD_SPEED]),
                "gear": int(driver_data_array[frame_idx, driver_idx, FIELD_GEAR]),
                "drs": int(driver_data_array[frame_idx, driver_idx, FIELD_DRS]),
            }
        
        frames.append({
            "t": float(frame_metadata[frame_idx, 0]),
            "lap": int(frame_metadata[frame_idx, 1]),
            "drivers": frame_data,
        })
    
    return frames


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
            # status['Time'] is a timedelta object from pandas
            seconds = status['Time'].total_seconds()
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
