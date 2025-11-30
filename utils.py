import fastf1
import pandas as pd
import streamlit as st

# 設定 Cache，避免每次都要重抓
fastf1.Cache.enable_cache("cache")


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
def process_replay_data(session):
    """
    這就是 f1-race-replay 的核心邏輯移植。
    將所有車手的遙測數據同步到同一個時間軸上，以便播放。
    """
    # 取得所有車手
    drivers = session.results["Abbreviation"].unique()

    # 建立一個巨大的 DataFrame，包含每一秒所有車手的位置
    # 為了效能，我們這裡做一點簡化，每 0.5 秒一個採樣點
    all_telemetry = []

    for driver in drivers:
        try:
            # 獲取該車手的遙測數據
            laps = session.laps.pick_driver(driver)
            # 這裡我們合併所有圈數的數據
            telemetry = laps.get_telemetry().add_distance()
            telemetry["Driver"] = driver

            # 保留關鍵欄位：時間、X、Y、速度、輪胎(需額外處理，這裡先略過以保效能)
            # 將時間轉換為總秒數 (SessionTime)
            telemetry["TimeSec"] = telemetry["SessionTime"].dt.total_seconds()

            # 降採樣 (Downsample) 以提升 Streamlit 播放效能
            # 每 10 筆資料取 1 筆 (FastF1 原本約 0.2秒一筆)
            telemetry = telemetry.iloc[::5, :]

            all_telemetry.append(
                telemetry[["TimeSec", "Driver", "X", "Y", "Speed", "RPM", "nGear"]]
            )
        except Exception:
            continue

    if not all_telemetry:
        return pd.DataFrame(), None

    big_df = pd.concat(all_telemetry)

    # 為了讓播放器好讀取，我們對齊時間軸
    # 找出比賽開始與結束時間
    start_time = big_df["TimeSec"].min()
    end_time = big_df["TimeSec"].max()

    return big_df, (start_time, end_time)


def simulate_strategy_change(original_laps, driver, pit_lap_offset, new_compound):
    """
    簡易模擬邏輯：如果晚進站，會發生什麼事？
    這是一個數學模型。
    """
    # 複製一份數據以免改到原始檔
    sim_laps = original_laps.copy()

    # 這裡實作簡單的物理邏輯
    # 假設：軟胎每圈快 0.5 秒，但每圈衰退 0.1 秒
    # 這部分可以根據你的需求寫得更複雜

    return sim_laps  # 回傳模擬後的數據
