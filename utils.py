import os
import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np
import streamlit as st

# 設定 Cache，避免每次都要重抓
CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
fastf1.Cache.enable_cache(CACHE_DIR)

# Frame rate for replay (from f1-race-replay)
FPS = 4
DT = 1 / FPS

# Tyre compound mapping (from f1-race-replay)
TYRE_COMPOUNDS = {
    "SOFT": 0,
    "MEDIUM": 1,
    "HARD": 2,
    "INTERMEDIATE": 3,
    "WET": 4,
}

# Track status mapping
TRACK_STATUS = {
    "1": "GREEN",
    "2": "YELLOW",
    "4": "SC",
    "5": "RED",
    "6": "VSC",
    "7": "VSC_ENDING",
}


def get_tyre_compound_int(compound_str):
    """Convert tyre compound string to integer."""
    if compound_str is None:
        return -1
    return TYRE_COMPOUNDS.get(str(compound_str).upper(), -1)


def get_tyre_compound_str(compound_int):
    """Convert tyre compound integer back to string."""
    for name, value in TYRE_COMPOUNDS.items():
        if value == compound_int:
            return name
    return "UNKNOWN"


@st.cache_data
def get_driver_colors(_session):
    """
    Get driver colors from FastF1 plotting module.
    Returns RGB tuples for each driver.
    """
    try:
        color_mapping = fastf1.plotting.get_driver_color_mapping(_session)
        rgb_colors = {}
        for driver, hex_color in color_mapping.items():
            hex_color = hex_color.lstrip("#")
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            rgb_colors[driver] = (r, g, b)
        return rgb_colors
    except Exception:
        return {}


@st.cache_data
def load_session_data(year, gp, session_type):
    """載入比賽數據，包含車手列表與賽道資訊"""
    try:
        session = fastf1.get_session(year, gp, session_type)
        session.load(telemetry=True, weather=False)
        return session
    except Exception:
        return None


@st.cache_data
def process_replay_data(_session):
    """
    基於 f1-race-replay 的核心邏輯。
    將所有車手的遙測數據同步到同一個時間軸上，以便播放。
    包含：位置、速度、檔位、DRS、輪胎、圈數、賽道距離等。
    """
    drivers = _session.results["Abbreviation"].unique()

    driver_data = {}
    global_t_min = None
    global_t_max = None

    # 1. 收集每個車手的遙測數據
    for driver in drivers:
        try:
            laps_driver = _session.laps.pick_drivers(driver)
            if laps_driver.empty:
                continue

            t_all = []
            x_all = []
            y_all = []
            race_dist_all = []
            lap_numbers = []
            tyre_compounds = []
            speed_all = []
            gear_all = []
            drs_all = []

            total_dist_so_far = 0.0

            # 迭代每一圈
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
                speed_lap = lap_tel["Speed"].to_numpy()
                gear_lap = lap_tel["nGear"].to_numpy()
                drs_lap = lap_tel["DRS"].to_numpy()

                # 標準化圈內距離從0開始
                d_lap = d_lap - d_lap.min()
                lap_length = d_lap.max()

                # 累積賽道距離
                race_d_lap = total_dist_so_far + d_lap
                total_dist_so_far += lap_length

                t_all.append(t_lap)
                x_all.append(x_lap)
                y_all.append(y_lap)
                race_dist_all.append(race_d_lap)
                lap_numbers.append(np.full_like(t_lap, lap_number))
                tyre_compounds.append(np.full_like(t_lap, tyre_compound_int))
                speed_all.append(speed_lap)
                gear_all.append(gear_lap)
                drs_all.append(drs_lap)

            if not t_all:
                continue

            # 合併所有圈的數據
            t_all = np.concatenate(t_all)
            x_all = np.concatenate(x_all)
            y_all = np.concatenate(y_all)
            race_dist_all = np.concatenate(race_dist_all)
            lap_numbers = np.concatenate(lap_numbers)
            tyre_compounds = np.concatenate(tyre_compounds)
            speed_all = np.concatenate(speed_all)
            gear_all = np.concatenate(gear_all)
            drs_all = np.concatenate(drs_all)

            # 按時間排序
            order = np.argsort(t_all)
            t_all = t_all[order]
            x_all = x_all[order]
            y_all = y_all[order]
            race_dist_all = race_dist_all[order]
            lap_numbers = lap_numbers[order]
            tyre_compounds = tyre_compounds[order]
            speed_all = speed_all[order]
            gear_all = gear_all[order]
            drs_all = drs_all[order]

            driver_data[driver] = {
                "t": t_all,
                "x": x_all,
                "y": y_all,
                "dist": race_dist_all,
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

        except (KeyError, ValueError, AttributeError):
            # Skip drivers with missing or invalid telemetry data
            continue

    if not driver_data:
        return pd.DataFrame(), None, [], {}

    # 2. 創建統一時間軸
    timeline = np.arange(global_t_min, global_t_max, DT) - global_t_min

    # 3. 重採樣每個車手的數據到統一時間軸
    resampled_data = {}
    for code, data in driver_data.items():
        t = data["t"] - global_t_min

        order = np.argsort(t)
        t_sorted = t[order]
        x_sorted = data["x"][order]
        y_sorted = data["y"][order]
        dist_sorted = data["dist"][order]
        lap_sorted = data["lap"][order]
        tyre_sorted = data["tyre"][order]
        speed_sorted = data["speed"][order]
        gear_sorted = data["gear"][order]
        drs_sorted = data["drs"][order]

        x_resampled = np.interp(timeline, t_sorted, x_sorted)
        y_resampled = np.interp(timeline, t_sorted, y_sorted)
        dist_resampled = np.interp(timeline, t_sorted, dist_sorted)
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
            "lap": lap_resampled,
            "tyre": tyre_resampled,
            "speed": speed_resampled,
            "gear": gear_resampled,
            "drs": drs_resampled,
        }

    # 4. 處理賽道狀態 (Track Status)
    track_statuses = []
    try:
        track_status = _session.track_status
        for status in track_status.to_dict("records"):
            seconds = status["Time"].total_seconds()
            start_time = seconds - global_t_min

            if track_statuses:
                track_statuses[-1]["end_time"] = start_time

            track_statuses.append(
                {
                    "status": status["Status"],
                    "status_name": TRACK_STATUS.get(
                        str(status["Status"]), str(status["Status"])
                    ),
                    "start_time": start_time,
                    "end_time": None,
                }
            )
    except (KeyError, AttributeError, TypeError):
        # Track status data may not be available for all sessions
        pass

    # 5. 構建幀數據 (Frame-based data)
    frames = []
    for i, t in enumerate(timeline):
        snapshot = []
        for code, d in resampled_data.items():
            snapshot.append(
                {
                    "code": code,
                    "dist": float(d["dist"][i]),
                    "x": float(d["x"][i]),
                    "y": float(d["y"][i]),
                    "lap": int(round(d["lap"][i])),
                    "tyre": int(d["tyre"][i]),
                    "speed": float(d["speed"][i]),
                    "gear": int(d["gear"][i]),
                    "drs": int(d["drs"][i]),
                }
            )

        if not snapshot:
            continue

        # 按賽道距離排序得到位置
        snapshot.sort(key=lambda r: r["dist"], reverse=True)

        leader = snapshot[0]
        leader_lap = leader["lap"]

        frame_data = {}
        for idx, car in enumerate(snapshot):
            code = car["code"]
            position = idx + 1

            # 計算與前車的差距
            gap_to_leader = leader["dist"] - car["dist"]

            frame_data[code] = {
                "x": car["x"],
                "y": car["y"],
                "dist": car["dist"],
                "lap": car["lap"],
                "tyre": car["tyre"],
                "tyre_name": get_tyre_compound_str(car["tyre"]),
                "position": position,
                "speed": car["speed"],
                "gear": car["gear"],
                "drs": car["drs"],
                "gap_to_leader": gap_to_leader,
            }

        frames.append(
            {
                "t": float(t),
                "lap": leader_lap,
                "drivers": frame_data,
            }
        )

    # 6. 轉換為 DataFrame 格式以供 Streamlit 使用
    all_rows = []
    for frame in frames:
        t = frame["t"]
        for code, data in frame["drivers"].items():
            all_rows.append(
                {
                    "TimeSec": t,
                    "Driver": code,
                    "X": data["x"],
                    "Y": data["y"],
                    "Speed": data["speed"],
                    "nGear": data["gear"],
                    "DRS": data["drs"],
                    "Lap": data["lap"],
                    "Tyre": data["tyre_name"],
                    "Position": data["position"],
                    "Distance": data["dist"],
                    "GapToLeader": data["gap_to_leader"],
                }
            )

    big_df = pd.DataFrame(all_rows)

    # 獲取車手顏色
    driver_colors = get_driver_colors(_session)

    return big_df, (0, timeline[-1]), track_statuses, driver_colors


def get_track_status_at_time(track_statuses, current_time):
    """Get the track status at a specific time."""
    for status in track_statuses:
        if status["start_time"] <= current_time:
            if status["end_time"] is None or current_time < status["end_time"]:
                return status["status_name"]
    return "GREEN"


def get_drs_status_text(drs_value):
    """Convert DRS integer value to status text."""
    if drs_value in [0, 1]:
        return "Off"
    elif drs_value == 8:
        return "Eligible"
    elif drs_value in [10, 12, 14]:
        return "Active"
    return "Unknown"


def simulate_strategy_change(original_laps, driver, pit_lap_offset, new_compound):
    """
    簡易模擬邏輯：如果晚進站，會發生什麼事？
    這是一個數學模型。

    輪胎衰退模型:
    - SOFT: base_delta=-0.5s (faster), degradation=0.15s/lap
    - MEDIUM: base_delta=0.0s (baseline), degradation=0.08s/lap
    - HARD: base_delta=+0.3s (slower), degradation=0.04s/lap
    """
    # 複製一份數據以免改到原始檔
    sim_laps = original_laps.copy()

    # 這裡實作簡單的物理邏輯
    # 假設：軟胎每圈快 0.5 秒，但每圈衰退更快
    # 這部分可以根據你的需求寫得更複雜

    return sim_laps  # 回傳模擬後的數據
