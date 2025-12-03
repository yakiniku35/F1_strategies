#!/usr/bin/env python
"""
F1 Race Prediction Simulator
============================

Interactive main entry point for F1 race prediction and simulation.

For historical race replays, please use: https://github.com/IAmTomShaw/f1-race-replay

Usage:
    python main.py                          # Interactive mode
    python main.py --predict --year 2025 --gp Monaco
    python main.py --schedule               # View race calendar
"""

import sys
import argparse
import logging
from tabulate import tabulate

# Configure logging to suppress verbose FastF1 output
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s'
)
# Suppress FastF1 and other library logs
logging.getLogger('fastf1').setLevel(logging.ERROR)  # Changed to ERROR to suppress INFO
logging.getLogger('fastf1.core').setLevel(logging.ERROR)
logging.getLogger('fastf1.req').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from src.f1_data import get_race_telemetry, get_race_data, load_race_session, enable_cache
from src.arcade_replay import run_arcade_replay
from src.simulation import PredictedRaceSimulator, FutureRaceDataProvider
from src.ml_predictor import PreRacePredictor
from src.replay_wrapper import replay_race_external  # External f1-race-replay integration


def print_banner():
    """Print the welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║           F1 Race Prediction Simulator 🏎️                ║
╠══════════════════════════════════════════════════════════╣
║  1. 🔮 預測未來比賽 (Predict Future Race)                ║
║  2. 📼 回放歷史比賽 (Replay Historical Race)             ║
║  3. 📅 查看賽程表 (View Schedule)                        ║
║  4. ❌ 離開 (Exit)                                       ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)


def get_user_choice(prompt, valid_choices):
    """Get a valid choice from the user."""
    while True:
        choice = input(prompt).strip()
        if choice in valid_choices:
            return choice
        print(f"無效選擇，請輸入 {'/'.join(valid_choices)}")


def get_year_input():
    """Get year input from user."""
    while True:
        try:
            year_str = input("\n年份 Year (例如 2025): ").strip()
            year = int(year_str)
            if 2018 <= year <= 2030:
                return year
            print("請輸入有效年份 (2018-2030)")
        except ValueError:
            print("請輸入有效數字")


def get_gp_input():
    """Get GP name input from user."""
    gp = input("大獎賽 Grand Prix (例如 Monaco): ").strip()
    if not gp:
        print("使用預設: Monaco")
        return "Monaco"
    return gp


def view_schedule():
    """Display the 2025 F1 schedule."""
    data_provider = FutureRaceDataProvider()
    schedule = data_provider.get_2025_schedule()

    print("\n📅 2025 F1 賽程表 (2025 F1 Schedule)")
    print("=" * 60)

    table_data = []
    for race in schedule:
        table_data.append([
            race['round'],
            race['name'],
            race['location'],
            race['date'],
            race['laps']
        ])

    headers = ['Round', 'Grand Prix', 'Location', 'Date', 'Laps']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))


def predict_future_race(year, gp, speed=1.0, train_model=True):
    """Run a future race prediction and simulation."""
    print(f"\n🔮 預測 {year} {gp} Grand Prix...")
    print("=" * 50)

    # Create simulator
    print("⏳ 正在初始化預測系統...")
    simulator = PredictedRaceSimulator(year, gp)

    # Get race info
    race_info = simulator.race_info
    if race_info:
        print(f"✅ 比賽: {race_info.get('name', gp)}")
        print(f"   地點: {race_info.get('location', 'Unknown')}")
        print(f"   圈數: {race_info.get('laps', 50)}")
    else:
        print(f"⚠️ 找不到 {gp} 的賽程資訊，使用預設值")

    # Train ML model if requested
    if train_model:
        print("\n⏳ 正在訓練預測模型（可能需要幾分鐘）...")
        predictor = PreRacePredictor()
        try:
            # Try to train on historical data (may fail if no network)
            predictor.train_on_historical_data([2023, 2024])
            print("✅ 模型訓練完成")
        except Exception as e:
            print(f"⚠️ 使用預設模型")

    # Get qualifying prediction
    print("\n⏳ 正在生成排位賽預測...")
    qualifying = simulator.get_qualifying_results()

    # Get prediction confidence
    confidences = simulator.get_prediction_confidence()

    # Display qualifying prediction
    print(f"\n🏁 {year} {gp} Grand Prix 預測結果：")

    table_data = []
    for quali in qualifying[:10]:  # Top 10
        conf = confidences.get(quali['code'], 0.7)
        table_data.append([
            quali['grid'],
            quali['code'],
            quali['name'],
            quali['team'],
            f"{conf * 100:.0f}%"
        ])

    headers = ['排名', '車手', '姓名', '車隊', '預測信心度']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

    # Ask user if they want to run simulation
    print("\n是否開啟賽道模擬視窗？(y/n): ", end="")
    run_sim = input().strip().lower()

    if run_sim in ['y', 'yes', '是']:
        print("\n⏳ 正在生成比賽模擬數據...")

        # Generate simulation frames
        sim_data = simulator.generate_simulated_frames()

        print(f"✅ 生成了 {len(sim_data['frames'])} 個模擬幀")
        print("\n🎬 開啟賽道模擬視窗...")

        # Run the visualization
        run_arcade_replay(
            frames=sim_data['frames'],
            track_statuses=sim_data['track_statuses'],
            example_lap=sim_data['example_lap'],
            drivers=sim_data['drivers'],
            playback_speed=speed,
            driver_colors=sim_data['driver_colors'],
            title=f"🔮 PREDICTED - {year} {gp} GP",
            mode='predicted',
            race_info={'year': year, 'gp': gp}
        )


def replay_historical_race(year, gp, speed=1.0, use_optimized=True):
    """Replay a historical race using external f1-race-replay system.
    
    Args:
        year: Race year
        gp: Grand Prix name or round number
        speed: Playback speed multiplier (not used in external system)
        use_optimized: Compatibility parameter (not used in external system)
    """
    # Use the external f1-race-replay system
    replay_race_external(year, gp, session_type='R')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='F1 Race Prediction Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                              # Interactive mode
    python main.py --predict --year 2025 --gp Monaco
    python main.py --replay --year 2024 --gp Monaco  # Uses f1-race-replay system
    python main.py --schedule

Replay system powered by: https://github.com/IAmTomShaw/f1-race-replay
        """
    )

    parser.add_argument('--predict', action='store_true',
                        help='Predict a future race')
    parser.add_argument('--replay', action='store_true',
                        help='Replay a historical race (uses external f1-race-replay)')
    parser.add_argument('--schedule', action='store_true',
                        help='View 2025 schedule')
    parser.add_argument('--year', type=int, default=None,
                        help='Race year')
    parser.add_argument('--gp', type=str, default=None,
                        help='Grand Prix name (e.g., Monaco, Silverstone)')
    parser.add_argument('--round', type=int, default=None,
                        help='Round number (alternative to --gp)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Initial playback speed (default: 1.0)')
    parser.add_argument('--no-train', action='store_true',
                        help='Skip ML model training')

    return parser.parse_args()


def interactive_mode():
    """Run in interactive mode."""
    print_banner()

    choice = get_user_choice("請選擇 (1-4): ", ['1', '2', '3', '4'])

    if choice == '1':
        # Predict future race
        year = get_year_input()
        gp = get_gp_input()
        predict_future_race(year, gp)

    elif choice == '2':
        # Replay historical race (using external f1-race-replay)
        year = get_year_input()
        gp = get_gp_input()
        replay_historical_race(year, gp)

    elif choice == '3':
        # View schedule
        view_schedule()
        input("\n按 Enter 返回主選單...")
        interactive_mode()

    elif choice == '4':
        # Exit
        print("\n再見！🏎️")
        sys.exit(0)


def main():
    """Main entry point."""
    args = parse_args()

    # Handle command line mode
    if args.schedule:
        view_schedule()
        return

    if args.predict:
        year = args.year or 2025
        gp = args.gp or args.round
        if gp is None:
            print("錯誤: 請指定 --gp 或 --round")
            sys.exit(1)
        predict_future_race(year, gp, args.speed, not args.no_train)
        return

    if args.replay:
        year = args.year or 2024
        gp = args.gp or args.round
        if gp is None:
            print("錯誤: 請指定 --gp 或 --round")
            sys.exit(1)
        replay_historical_race(year, gp, args.speed)
        return

    # If no mode specified, run interactive mode
    interactive_mode()


if __name__ == "__main__":
    main()
