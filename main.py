#!/usr/bin/env python
"""
F1 Race Replay with ML Prediction
Main entry point for the Arcade-based F1 race visualization.

Usage:
    python main.py --year 2023 --gp Monaco
    python main.py --year 2023 --round 7
    python main.py --year 2023 --gp Monaco --refresh-data
"""

import sys
import argparse
from src.f1_data import get_race_telemetry, load_race_session, enable_cache
from src.arcade_replay import run_arcade_replay


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='F1 Race Replay with ML Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --year 2023 --gp Monaco
    python main.py --year 2023 --round 7
    python main.py --year 2023 --gp "Saudi Arabian" --session R
    python main.py --year 2024 --round 1 --refresh-data
        """
    )

    parser.add_argument('--year', type=int, default=2023,
                        help='Race year (default: 2023)')
    parser.add_argument('--gp', type=str, default=None,
                        help='Grand Prix name (e.g., Monaco, Silverstone)')
    parser.add_argument('--round', type=int, default=None,
                        help='Round number (alternative to --gp)')
    parser.add_argument('--session', type=str, default='R',
                        choices=['R', 'Q', 'FP1', 'FP2', 'FP3', 'S', 'SS'],
                        help='Session type: R=Race, Q=Qualifying (default: R)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Initial playback speed (default: 1.0)')
    parser.add_argument('--refresh-data', action='store_true',
                        help='Force refresh of telemetry data')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate arguments
    if args.gp is None and args.round is None:
        print("Error: Please specify either --gp or --round")
        print("Example: python main.py --year 2023 --gp Monaco")
        print("         python main.py --year 2023 --round 7")
        sys.exit(1)

    # Enable FastF1 cache
    enable_cache()

    # Load race session
    race_identifier = args.gp if args.gp else args.round
    print(f"Loading {args.year} {race_identifier} {args.session}...")

    try:
        session = load_race_session(args.year, race_identifier, args.session)
        event_name = session.event['EventName']
        print(f"Loaded session: {event_name} - Round {session.event['RoundNumber']}")
    except Exception as e:
        print(f"Error loading session: {e}")
        print("\nTip: Check the year, GP name or round number.")
        print("Common GP names: Monaco, Silverstone, Monza, Spa, Suzuka, etc.")
        sys.exit(1)

    # Get race telemetry
    print("Processing telemetry data...")
    race_telemetry = get_race_telemetry(session, refresh_data=args.refresh_data)

    if not race_telemetry['frames']:
        print("Error: No telemetry data available for this session")
        sys.exit(1)

    # Get example lap for track layout
    try:
        example_lap = session.laps.pick_fastest().get_telemetry()
    except Exception:
        # Fallback to first available lap
        example_lap = session.laps.iloc[0].get_telemetry()

    # Get drivers list
    drivers = [session.get_driver(num)["Abbreviation"] for num in session.drivers]

    print(f"\nStarting replay for {event_name}")
    print(f"Drivers: {len(drivers)}")
    print(f"Total frames: {len(race_telemetry['frames'])}")
    print(f"Playback speed: {args.speed}x")
    print("\nOpening replay window...")

    # Run the replay
    run_arcade_replay(
        frames=race_telemetry['frames'],
        track_statuses=race_telemetry['track_statuses'],
        example_lap=example_lap,
        drivers=drivers,
        playback_speed=args.speed,
        driver_colors=race_telemetry['driver_colors'],
        title=f"{event_name} - {args.session} - F1 Replay with ML"
    )


if __name__ == "__main__":
    main()
