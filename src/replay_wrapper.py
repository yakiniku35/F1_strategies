"""
Wrapper for external f1-race-replay system
Integrates IAmTomShaw's replay into this project
https://github.com/IAmTomShaw/f1-race-replay
"""

from src.external_f1_data import get_race_telemetry, load_race_session, enable_cache, get_circuit_rotation
from src.external_replay import run_arcade_replay


def replay_race_external(year, gp, session_type='R'):
    """
    使用外部 f1-race-replay 系統回放比賽
    
    Args:
        year: 年份
        gp: 大獎賽名稱或回合數
        session_type: 'R' (Race) 或 'S' (Sprint)
    """
    print(f"\n📼 載入 {year} {gp} 歷史比賽 (使用 f1-race-replay 系統)...")
    print("==================================================")
    
    # Enable cache
    enable_cache()
    
    # Load session
    try:
        # Try to parse as round number first
        try:
            round_number = int(gp)
        except (ValueError, TypeError):
            # If not a number, it's a GP name - need to find round number
            import fastf1
            schedule = fastf1.get_event_schedule(year)
            
            # Try exact match first
            match = schedule[schedule['EventName'].str.contains(gp, case=False, na=False)]
            if match.empty:
                # Try partial match
                match = schedule[schedule['Location'].str.contains(gp, case=False, na=False)]
            
            if match.empty:
                raise ValueError(f"找不到 {year} 年的 '{gp}' 大獎賽")
            
            round_number = int(match.iloc[0]['RoundNumber'])
        
        session = load_race_session(year, round_number, session_type)
        print(f"✅ 載入成功: {session.event['EventName']} - Round {session.event['RoundNumber']}")
        
    except Exception as e:
        print(f"❌ 載入比賽失敗: {e}")
        return
    
    # Get race telemetry
    print("⏳ 正在處理遙測數據...")
    try:
        race_telemetry = get_race_telemetry(session, session_type=session_type)
    except Exception as e:
        print(f"❌ 處理遙測數據失敗: {e}")
        return
    
    # Get example lap for track layout
    try:
        example_lap = session.laps.pick_fastest().get_telemetry()
    except Exception:
        try:
            example_lap = session.laps.iloc[0].get_telemetry()
        except Exception as e:
            print(f"❌ 無法獲取賽道布局: {e}")
            return
    
    # Get drivers
    drivers = session.drivers
    
    # Get circuit rotation
    try:
        circuit_rotation = get_circuit_rotation(session)
    except Exception:
        circuit_rotation = 0
    
    print(f"\n🏁 準備開始回放")
    print(f"   車手數量: {len(drivers)}")
    print(f"   總幀數: {len(race_telemetry['frames']):,}")
    print("\n🎬 開啟回放視窗...")
    
    # Run the arcade replay
    try:
        run_arcade_replay(
            frames=race_telemetry['frames'],
            track_statuses=race_telemetry['track_statuses'],
            example_lap=example_lap,
            drivers=drivers,
            playback_speed=1.0,
            driver_colors=race_telemetry['driver_colors'],
            title=f"{session.event['EventName']} - {'Sprint' if session_type == 'S' else 'Race'}",
            total_laps=race_telemetry.get('total_laps'),
            circuit_rotation=circuit_rotation,
        )
    except Exception as e:
        print(f"❌ 回放失敗: {e}")
        import traceback
        traceback.print_exc()
