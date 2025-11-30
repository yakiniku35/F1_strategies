import os

import fastf1


CACHE_DIR = 'cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

fastf1.Cache.enable_cache(CACHE_DIR)


def load_race_data(year, grandprix, session_type='R'):
    """
    載入指定比賽的數據
    Load data for the specified race
    """
    try:
        session = fastf1.get_session(year, grandprix, session_type)
        session.load(telemetry=True, weather=True)
        return session
    except Exception as e:
        return str(e)


def get_driver_lap_data(session, driver_code):
    """
    載入特定車手的所有圈速數據
    Load all lap data for a specific driver
    """
    laps = session.laps.pick_driver(driver_code)
    # 挑選乾淨的圈（沒有進站、沒有黃旗）作為基準分析
    clean_laps = laps.clean(clean=True)
    return laps, clean_laps
