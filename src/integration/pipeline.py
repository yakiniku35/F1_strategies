"""
F1 Integration Pipeline Module.
Connects ML prediction, race simulation, and data table generation.
"""

import os
from typing import Optional


class F1IntegrationPipeline:
    """
    Integration pipeline that connects prediction, simulation, and table display.

    This class manages the complete workflow from loading race data,
    running ML predictions, to generating visualizations and tables.
    """

    def __init__(self, year: int, gp: str, session_type: str = 'R'):
        """
        Initialize the integration pipeline.

        Args:
            year: Race year (e.g., 2024)
            gp: Grand Prix name (e.g., 'Monaco')
            session_type: Session type ('R' for Race, 'Q' for Qualifying, etc.)
        """
        self.year = year
        self.gp = gp
        self.session_type = session_type
        self.session = None
        self.race_telemetry = None
        self.predictions = {}
        self.tables = {}
        self.ml_predictor = None

    def load_data(self, refresh_data: bool = False) -> bool:
        """
        Load race session and telemetry data.

        Args:
            refresh_data: If True, force reload of telemetry data

        Returns:
            True if data loaded successfully, False otherwise
        """
        from src.f1_data import load_race_session, get_race_telemetry, enable_cache

        enable_cache()

        try:
            print(f"Loading {self.year} {self.gp} {self.session_type}...")
            self.session = load_race_session(self.year, self.gp, self.session_type)

            event_name = self.session.event['EventName']
            print(f"Loaded session: {event_name} - Round {self.session.event['RoundNumber']}")

            print("Processing telemetry data...")
            self.race_telemetry = get_race_telemetry(self.session, refresh_data=refresh_data)

            if not self.race_telemetry['frames']:
                print("Error: No telemetry data available")
                return False

            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def run_prediction(self) -> dict:
        """
        Run ML predictions on the race data.

        Returns:
            Dictionary containing predictions for all drivers
        """
        from src.ml_predictor import RaceTrendPredictor

        if not self.race_telemetry:
            print("Error: No race data loaded. Call load_data() first.")
            return {}

        frames = self.race_telemetry['frames']
        drivers = list(frames[0]['drivers'].keys()) if frames else []

        print("Training ML prediction model...")
        self.ml_predictor = RaceTrendPredictor()

        if not self.ml_predictor.train(frames, drivers):
            print("Warning: ML training failed")
            return {}

        # Generate predictions for the latest frame
        if frames:
            latest_frame = frames[-1]
            self.predictions = self.ml_predictor.predict_all_drivers(latest_frame)

            # Add pit strategy analysis
            for driver_code in self.predictions:
                driver_data = latest_frame['drivers'].get(driver_code, {})
                current_lap = driver_data.get('lap', 1)
                pit_analysis = self.ml_predictor.analyze_pit_strategy(
                    frames, driver_code, current_lap
                )
                if pit_analysis:
                    self.predictions[driver_code]['pit_strategy'] = pit_analysis

        print(f"Generated predictions for {len(self.predictions)} drivers")
        return self.predictions

    def run_simulation_with_predictions(self, playback_speed: float = 1.0):
        """
        Run the race simulation with prediction overlays.

        Args:
            playback_speed: Playback speed multiplier
        """
        from src.arcade_replay import run_arcade_replay

        if not self.race_telemetry or not self.session:
            print("Error: No race data loaded. Call load_data() first.")
            return

        # Get example lap for track layout
        try:
            example_lap = self.session.laps.pick_fastest().get_telemetry()
        except Exception:
            example_lap = self.session.laps.iloc[0].get_telemetry()

        drivers = [
            self.session.get_driver(num)["Abbreviation"]
            for num in self.session.drivers
        ]

        event_name = self.session.event['EventName']
        print(f"\nStarting replay for {event_name}")
        print(f"Drivers: {len(drivers)}")
        print(f"Total frames: {len(self.race_telemetry['frames'])}")
        print(f"Predictions loaded: {len(self.predictions)}")

        run_arcade_replay(
            frames=self.race_telemetry['frames'],
            track_statuses=self.race_telemetry['track_statuses'],
            example_lap=example_lap,
            drivers=drivers,
            playback_speed=playback_speed,
            driver_colors=self.race_telemetry['driver_colors'],
            title=f"{event_name} - {self.session_type} - F1 Replay with ML",
            predictions=self.predictions
        )

    def generate_tables(self, output_dir: Optional[str] = None) -> dict:
        """
        Generate data tables from predictions.

        Args:
            output_dir: Directory to export tables (optional)

        Returns:
            Dictionary containing all generated tables
        """
        from src.dashboard.tables import (
            generate_driver_standings_table,
            generate_strategy_table,
            generate_battle_table,
            export_to_csv,
            export_to_json,
        )

        if not self.race_telemetry:
            print("Error: No race data loaded. Call load_data() first.")
            return {}

        frames = self.race_telemetry['frames']
        if not frames:
            return {}

        latest_frame = frames[-1]

        # Generate tables
        self.tables['driver_standings'] = generate_driver_standings_table(
            latest_frame, self.predictions
        )
        self.tables['strategy'] = generate_strategy_table(
            latest_frame, self.predictions
        )
        self.tables['battles'] = generate_battle_table(
            latest_frame, self.predictions
        )

        # Print tables to console
        print("\n" + "=" * 60)
        print("DRIVER STANDINGS PREDICTIONS")
        print("=" * 60)
        print(self.tables['driver_standings'])

        print("\n" + "=" * 60)
        print("STRATEGY RECOMMENDATIONS")
        print("=" * 60)
        print(self.tables['strategy'])

        print("\n" + "=" * 60)
        print("BATTLE PREDICTIONS")
        print("=" * 60)
        print(self.tables['battles'])

        # Export to files if output_dir specified
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            event_name = self.session.event['EventName'].replace(' ', '_')

            # Export CSV files
            for table_name, table_data in self.tables.items():
                csv_path = os.path.join(output_dir, f"{event_name}_{table_name}.csv")
                export_to_csv(table_data, csv_path)

            # Export all predictions to JSON
            json_path = os.path.join(output_dir, f"{event_name}_predictions.json")
            export_to_json(self.predictions, json_path)

            print(f"\nTables exported to: {output_dir}")

        return self.tables

    def run_full_pipeline(self, output_dir: Optional[str] = None,
                          playback_speed: float = 1.0,
                          skip_simulation: bool = False):
        """
        Execute the complete integration pipeline.

        Args:
            output_dir: Directory for exported tables
            playback_speed: Simulation playback speed
            skip_simulation: If True, skip the visual simulation
        """
        print("=" * 60)
        print("F1 INTEGRATION PIPELINE")
        print("=" * 60)

        # Step 1: Load data
        print("\n[Step 1/4] Loading race data...")
        if not self.load_data():
            print("Pipeline failed: Could not load race data")
            return

        # Step 2: Run predictions
        print("\n[Step 2/4] Running ML predictions...")
        self.run_prediction()

        # Step 3: Generate tables
        print("\n[Step 3/4] Generating data tables...")
        self.generate_tables(output_dir=output_dir)

        # Step 4: Run simulation
        if not skip_simulation:
            print("\n[Step 4/4] Starting race simulation...")
            self.run_simulation_with_predictions(playback_speed=playback_speed)
        else:
            print("\n[Step 4/4] Simulation skipped")

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
