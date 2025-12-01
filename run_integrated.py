#!/usr/bin/env python
"""
F1 Integrated Race Analysis
Main entry point for the integrated F1 analysis pipeline.

This script combines ML prediction, race simulation, and data table generation
into a unified workflow.

Usage:
    python run_integrated.py --year 2024 --gp Monaco --mode full
    python run_integrated.py --year 2024 --gp Monaco --mode predict-only
    python run_integrated.py --year 2024 --gp Monaco --mode tables-only
    python run_integrated.py --year 2024 --gp Monaco --mode simulation-only
"""

import sys
import argparse

from src.integration.pipeline import F1IntegrationPipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='F1 Integrated Race Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
    full            - Run complete pipeline (predict + tables + simulation)
    predict-only    - Run ML predictions only
    tables-only     - Generate data tables only (requires predictions)
    simulation-only - Run simulation only

Examples:
    python run_integrated.py --year 2023 --gp Monaco --mode full
    python run_integrated.py --year 2024 --round 7 --mode predict-only
    python run_integrated.py --year 2023 --gp "Saudi Arabian" --mode tables-only --output ./output
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
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'predict-only', 'tables-only', 'simulation-only'],
                        help='Execution mode (default: full)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Simulation playback speed (default: 1.0)')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory for tables (default: ./output)')
    parser.add_argument('--refresh-data', action='store_true',
                        help='Force refresh of telemetry data')

    return parser.parse_args()


def main():
    """Main entry point for the integrated pipeline."""
    args = parse_args()

    # Validate arguments
    if args.gp is None and args.round is None:
        print("Error: Please specify either --gp or --round")
        print("Example: python run_integrated.py --year 2023 --gp Monaco --mode full")
        print("         python run_integrated.py --year 2023 --round 7 --mode predict-only")
        sys.exit(1)

    # Determine race identifier
    race_identifier = args.gp if args.gp else args.round

    # Initialize pipeline
    pipeline = F1IntegrationPipeline(
        year=args.year,
        gp=race_identifier,
        session_type=args.session
    )

    try:
        if args.mode == 'full':
            # Run complete pipeline
            pipeline.run_full_pipeline(
                output_dir=args.output,
                playback_speed=args.speed,
                skip_simulation=False
            )

        elif args.mode == 'predict-only':
            # Load data and run predictions only
            if not pipeline.load_data(refresh_data=args.refresh_data):
                print("Failed to load race data")
                sys.exit(1)

            predictions = pipeline.run_prediction()
            if predictions:
                print(f"\nGenerated predictions for {len(predictions)} drivers:")
                for driver, pred in predictions.items():
                    trend = pred.get('trend', 'stable')
                    pos = pred.get('current_position', '?')
                    pred_pos = pred.get('predicted_position', '?')
                    print(f"  {driver}: P{pos} -> P{pred_pos} ({trend})")

        elif args.mode == 'tables-only':
            # Load data, run predictions, and generate tables (no simulation)
            if not pipeline.load_data(refresh_data=args.refresh_data):
                print("Failed to load race data")
                sys.exit(1)

            pipeline.run_prediction()
            pipeline.generate_tables(output_dir=args.output)

        elif args.mode == 'simulation-only':
            # Load data and run simulation only
            if not pipeline.load_data(refresh_data=args.refresh_data):
                print("Failed to load race data")
                sys.exit(1)

            # Optionally run predictions for overlay
            pipeline.run_prediction()

            # Run simulation
            pipeline.run_simulation_with_predictions(playback_speed=args.speed)

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nPipeline error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
