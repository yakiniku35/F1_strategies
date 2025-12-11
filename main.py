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
from src.strategy_analyzer import StrategyAnalyzer


def print_banner():
    """Print the welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║           F1 Race Prediction Simulator 🏎️                ║
╠══════════════════════════════════════════════════════════╣
║  1. 🔮 預測未來比賽 (Predict Future Race)                ║
║  2. 📼 回放歷史比賽 (Replay Historical Race)             ║
║  3. 📅 查看賽程表 (View Schedule)                        ║
║  4. 🎯 策略分析 (Strategy Analysis)                      ║
║  5. ❌ 離開 (Exit)                                       ║
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
    
    # Show strategy recommendations
    print("\n🎯 推薦策略 (Recommended Strategies):")
    print("=" * 50)
    
    # Get strategy options for a top driver
    if len(qualifying) > 0:
        top_driver = qualifying[0]
        print(f"\n針對 {top_driver['name']} ({top_driver['code']}) - P{top_driver['grid']}:")
        try:
            strategy_comparison = simulator.get_strategy_comparison(
                top_driver['code'], 
                top_driver['grid']
            )
            print(strategy_comparison)
        except Exception as e:
            print(f"⚠️ 策略分析暫時無法使用")
    
    print("\n💡 提示: 使用 'python main.py --strategy' 進行詳細策略分析")

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


def analyze_race_strategy():
    """Interactive race strategy analysis."""
    print("\n" + "=" * 60)
    print("🎯 F1 Strategy Analysis Tool")
    print("=" * 60)
    
    # Get track input
    print("\n可用賽道 (Available Tracks):")
    tracks = ["Monaco", "Silverstone", "Monza", "Spa", "Suzuka", "Barcelona", "Bahrain", "Singapore"]
    for i, track in enumerate(tracks, 1):
        print(f"  {i}. {track}")
    
    track_choice = input("\n選擇賽道 (Enter number or name): ").strip()
    
    # Parse track choice
    if track_choice.isdigit() and 1 <= int(track_choice) <= len(tracks):
        track_name = tracks[int(track_choice) - 1]
    else:
        track_name = track_choice if track_choice else "Silverstone"
    
    # Get total laps
    while True:
        try:
            laps_str = input("比賽圈數 Total laps (預設 50): ").strip()
            total_laps = int(laps_str) if laps_str else 50
            if 20 <= total_laps <= 80:
                break
            print("請輸入 20-80 之間的圈數")
        except ValueError:
            print("請輸入有效數字")
    
    # Initialize strategy analyzer
    analyzer = StrategyAnalyzer(track_name=track_name, total_laps=total_laps)
    
    print(f"\n📍 賽道: {track_name}")
    print(f"🏁 圈數: {total_laps}")
    
    # Strategy analysis menu
    while True:
        print("\n" + "-" * 60)
        print("策略分析選項 (Strategy Analysis Options):")
        print("  1. 比較策略選項 (Compare Strategy Options)")
        print("  2. Undercut 分析 (Undercut Analysis)")
        print("  3. Overcut 分析 (Overcut Analysis)")
        print("  4. 燃油策略 (Fuel Strategy)")
        print("  5. 完整策略報告 (Full Strategy Report)")
        print("  6. 返回主選單 (Back to Main Menu)")
        
        analysis_choice = input("\n選擇 (1-6): ").strip()
        
        if analysis_choice == '1':
            # Compare strategies
            print("\n⏳ 正在生成策略選項...")
            
            position_str = input("當前位置 Current position (預設 10): ").strip()
            current_position = int(position_str) if position_str else 10
            
            strategies = analyzer.generate_strategy_options(current_position)
            comparison = analyzer.compare_strategies(strategies)
            print(comparison)
            
            # Ask if user wants to export
            export_choice = input("\n匯出策略? Export strategies? (y=JSON/c=CSV/n=No): ").strip().lower()
            if export_choice == 'y':
                filename = f"strategy_{track_name}_{total_laps}laps.json"
                analyzer.export_strategies_to_json(strategies, filename)
            elif export_choice == 'c':
                filename = f"strategy_{track_name}_{total_laps}laps.csv"
                analyzer.export_strategies_to_csv(strategies, filename)
            
        elif analysis_choice == '2':
            # Undercut analysis
            print("\n🔍 Undercut 機會分析")
            
            try:
                current_lap = int(input("當前圈數 Current lap: ").strip() or 20)
                gap = float(input("與前車差距(秒) Gap to car ahead (s): ").strip() or 3.0)
                our_age = int(input("我方輪胎圈齡 Our tyre age (laps): ").strip() or 10)
                their_age = int(input("對手輪胎圈齡 Their tyre age (laps): ").strip() or 15)
                
                result = analyzer.analyze_undercut_opportunity(
                    current_lap, gap, our_age, their_age
                )
                
                print(f"\n{'='*60}")
                print(f"Undercut 可行性: {'🟢 可行' if result['viable'] else '🔴 不可行'}")
                print(f"評分: {result['score']:.2f} / 1.00")
                print(f"差距優勢: {'✓' if result['gap_advantage'] else '✗'}")
                print(f"輪胎優勢: {'✓' if result['tyre_advantage'] else '✗'}")
                print(f"進站窗口: {'✓' if result['in_pit_window'] else '✗'}")
                print(f"\n建議: {result['recommendation']}")
                print(f"{'='*60}")
                
            except ValueError:
                print("❌ 輸入無效")
                
        elif analysis_choice == '3':
            # Overcut analysis
            print("\n🔍 Overcut 機會分析")
            
            try:
                current_lap = int(input("當前圈數 Current lap: ").strip() or 25)
                gap = float(input("與前車差距(秒) Gap to car ahead (s): ").strip() or 5.0)
                our_age = int(input("我方輪胎圈齡 Our tyre age (laps): ").strip() or 15)
                compound = input("當前輪胎 Current compound (SOFT/MEDIUM/HARD): ").strip().upper() or "MEDIUM"
                
                result = analyzer.analyze_overcut_opportunity(
                    current_lap, gap, our_age, compound
                )
                
                print(f"\n{'='*60}")
                print(f"Overcut 可行性: {'🟢 可行' if result['viable'] else '🔴 不可行'}")
                print(f"剩餘最佳圈數: {result['remaining_optimal_laps']} laps")
                print(f"輪胎狀態: {'✓ 健康' if result['tyres_healthy'] else '✗ 需更換'}")
                print(f"建議延長: {result['recommended_extend_laps']} laps")
                print(f"\n建議: {result['recommendation']}")
                print(f"{'='*60}")
                
            except ValueError:
                print("❌ 輸入無效")
                
        elif analysis_choice == '4':
            # Fuel strategy
            print("\n⛽ 燃油策略模擬")
            
            try:
                fuel = float(input("起始燃油 Starting fuel (kg, 預設 110): ").strip() or 110)
                result = analyzer.simulate_fuel_strategy(fuel)
                
                print(f"\n{'='*60}")
                print(f"每圈油耗: {result['fuel_per_lap']:.2f} kg")
                print(f"起始時間損失: {result['initial_penalty']:.2f} s/lap")
                print(f"終盤時間損失: {result['final_penalty']:.2f} s/lap")
                print(f"最佳表現窗口: Lap {result['best_performance_window'][0]}-{result['best_performance_window'][1]}")
                print(f"\n💡 策略提示:")
                print(f"   {result['strategy_tip']}")
                print(f"{'='*60}")
                
                show_detail = input("\n顯示逐圈數據? Show lap-by-lap? (y/n): ").strip().lower()
                if show_detail == 'y':
                    print("\n圈數 | 燃油(kg) | 圈速(s) | 燃油損失(s)")
                    print("-" * 45)
                    for lap_data in result['lap_times'][::5]:  # Every 5 laps
                        print(f"L{lap_data['lap']:2d}  | {lap_data['fuel_kg']:6.1f}  | {lap_data['lap_time']:6.2f} | {lap_data['fuel_penalty']:6.2f}")
                        
            except ValueError:
                print("❌ 輸入無效")
                
        elif analysis_choice == '5':
            # Full strategy report
            print("\n📊 完整策略報告")
            
            try:
                current_lap = int(input("當前圈數 Current lap: ").strip() or 20)
                position = int(input("當前位置 Current position: ").strip() or 10)
                tyre = input("當前輪胎 Current tyre (SOFT/MEDIUM/HARD): ").strip().upper() or "MEDIUM"
                age = int(input("輪胎圈齡 Tyre age (laps): ").strip() or 10)
                
                summary = analyzer.get_strategy_summary(current_lap, position, tyre, age)
                
                print(f"\n{'='*60}")
                print(f"當前狀況 (Current Situation):")
                print(f"  圈數: Lap {summary['current_situation']['lap']}/{total_laps}")
                print(f"  位置: P{summary['current_situation']['position']}")
                print(f"  輪胎: {summary['current_situation']['tyre']} ({summary['current_situation']['tyre_age']} laps)")
                print(f"  輪胎健康: {summary['current_situation']['tyre_health'] * 100:.0f}%")
                print(f"  進站緊急度: {summary['current_situation']['pit_urgency']}")
                print(f"  最佳進站前圈數: {summary['current_situation']['laps_until_optimal_pit']} laps")
                print(f"\n下一步行動:")
                print(f"  {summary['next_action']}")
                
                print(f"\n推薦策略 (Top 3):")
                for i, strat in enumerate(summary['recommended_strategies'], 1):
                    print(f"\n  {i}. {strat.name}")
                    print(f"     進站: {strat.stops}次 | 風險: {strat.risk_level}")
                    print(f"     輪胎: {' → '.join(strat.compounds)}")
                    print(f"     預估時間: {strat.estimated_time:.0f}s")
                
                print(f"{'='*60}")
                
            except ValueError:
                print("❌ 輸入無效")
                
        elif analysis_choice == '6':
            # Back to main menu
            print("\n返回主選單...")
            interactive_mode()
            break
            
        else:
            print("無效選擇")
        
        input("\n按 Enter 繼續...")


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
    parser.add_argument('--strategy', action='store_true',
                        help='Run strategy analysis tool')
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
    parser.add_argument('--track', type=str, default=None,
                        help='Track name for strategy analysis')
    parser.add_argument('--laps', type=int, default=50,
                        help='Total laps for strategy analysis')

    return parser.parse_args()


def interactive_mode():
    """Run in interactive mode."""
    print_banner()

    choice = get_user_choice("請選擇 (1-5): ", ['1', '2', '3', '4', '5'])

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
        # Strategy analysis
        analyze_race_strategy()

    elif choice == '5':
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

    if args.strategy:
        # Quick strategy analysis from command line
        track = args.track or "Silverstone"
        laps = args.laps
        
        print(f"\n🎯 策略分析: {track} ({laps} laps)")
        print("=" * 60)
        
        analyzer = StrategyAnalyzer(track_name=track, total_laps=laps)
        strategies = analyzer.generate_strategy_options(current_position=10)
        comparison = analyzer.compare_strategies(strategies)
        print(comparison)
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
