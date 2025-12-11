"""
Machine Learning module for F1 race trend prediction.
Uses historical race data to predict future race outcomes.

Optimized to work directly with NumPy arrays for better performance.
"""

import numpy as np
import pickle
import os
import hashlib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# Import field indices for NumPy array access
try:
    from src.f1_data import (
        FIELD_X, FIELD_Y, FIELD_DIST, FIELD_REL_DIST, FIELD_LAP,
        FIELD_TYRE, FIELD_SPEED, FIELD_GEAR, FIELD_DRS, FIELD_POSITION
    )
except ImportError:
    # Fallback if running standalone
    FIELD_X, FIELD_Y, FIELD_DIST, FIELD_REL_DIST = 0, 1, 2, 3
    FIELD_LAP, FIELD_TYRE, FIELD_SPEED, FIELD_GEAR = 4, 5, 6, 7
    FIELD_DRS, FIELD_POSITION = 8, 9


class RaceTrendPredictor:
    """
    ML-based race trend predictor that analyzes race data
    and predicts future positions, lap times, and pit stop strategies.
    
    Supports both legacy frame format and optimized NumPy arrays.
    """

    def __init__(self, cache_dir: str = "cache/ml_models"):
        self.position_model = None
        self.laptime_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.predictions = {}
        self.prediction_history = []
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def prepare_training_data(self, frames, driver_code):
        """
        Prepare training data from race frames for a specific driver.
        Features: current_position, lap_number, tyre_compound, speed, gear, drs, distance
        Target: next_position (after N frames)
        """
        features = []
        targets_position = []
        targets_laptime = []

        # Look-ahead frames for prediction
        look_ahead = 250  # ~10 seconds at 25 FPS

        for i, frame in enumerate(frames[:-look_ahead]):
            if driver_code not in frame['drivers']:
                continue

            driver_data = frame['drivers'][driver_code]
            future_frame = frames[i + look_ahead]

            if driver_code not in future_frame['drivers']:
                continue

            future_data = future_frame['drivers'][driver_code]

            # Extract features
            feature_vector = [
                driver_data.get('position', 10),
                frame.get('lap', 1),
                driver_data.get('tyre', 1),
                driver_data.get('speed', 200),
                driver_data.get('gear', 5),
                1 if driver_data.get('drs', 0) in [10, 12, 14] else 0,
                driver_data.get('dist', 0),
                driver_data.get('rel_dist', 0),
            ]

            features.append(feature_vector)
            targets_position.append(future_data.get('position', 10))
            targets_laptime.append(future_data.get('speed', 200))

        return np.array(features), np.array(targets_position), np.array(targets_laptime)

    def prepare_training_data_numpy(self, driver_data_array, frame_metadata, driver_idx):
        """
        Prepare training data directly from NumPy arrays for a specific driver.
        This is significantly faster than the dictionary-based approach.
        
        Args:
            driver_data_array: NumPy 3D array (n_frames, n_drivers, n_fields)
            frame_metadata: NumPy 2D array (n_frames, 2) with [time, leader_lap]
            driver_idx: Index of the driver in the array
            
        Returns:
            Tuple of (features, position_targets, speed_targets) arrays
        """
        n_frames = driver_data_array.shape[0]
        look_ahead = 250  # ~10 seconds at 25 FPS
        
        if n_frames <= look_ahead:
            return np.array([]), np.array([]), np.array([])
        
        # Vectorized feature extraction
        # Current frame indices
        current_indices = np.arange(n_frames - look_ahead)
        future_indices = current_indices + look_ahead
        
        # Extract current driver data for all frames at once
        current_data = driver_data_array[current_indices, driver_idx, :]
        future_data = driver_data_array[future_indices, driver_idx, :]
        current_metadata = frame_metadata[current_indices, :]
        
        # Build feature matrix: [position, lap, tyre, speed, gear, drs_active, dist, rel_dist]
        # DRS values in FastF1: 0/1=Off, 8=Eligible/Ready, 10/12/14=Active (different states)
        drs_active = np.where(
            np.isin(current_data[:, FIELD_DRS].astype(int), [10, 12, 14]),
            1.0, 0.0
        )
        
        features = np.column_stack([
            current_data[:, FIELD_POSITION],
            current_metadata[:, 1],  # leader_lap (proxy for lap number)
            current_data[:, FIELD_TYRE],
            current_data[:, FIELD_SPEED],
            current_data[:, FIELD_GEAR],
            drs_active,
            current_data[:, FIELD_DIST],
            current_data[:, FIELD_REL_DIST],
        ])
        
        # Targets
        targets_position = future_data[:, FIELD_POSITION]
        targets_speed = future_data[:, FIELD_SPEED]
        
        return features, targets_position, targets_speed

    def train_from_numpy(self, driver_data_array, frame_metadata, driver_codes):
        """
        Train the prediction models using NumPy arrays directly.
        This is significantly faster than the dictionary-based train() method.
        
        Args:
            driver_data_array: NumPy 3D array (n_frames, n_drivers, n_fields)
            frame_metadata: NumPy 2D array (n_frames, 2) with [time, leader_lap]
            driver_codes: List of driver codes
            
        Returns:
            True if training succeeded, False otherwise
        """
        n_frames = driver_data_array.shape[0]
        n_drivers = driver_data_array.shape[1]
        
        if n_frames < 500:
            print("Not enough frames for ML training (need at least 500)")
            return False
        
        all_features = []
        all_position_targets = []
        all_speed_targets = []
        
        for driver_idx in range(n_drivers):
            features, pos_targets, speed_targets = self.prepare_training_data_numpy(
                driver_data_array, frame_metadata, driver_idx
            )
            if len(features) > 0:
                all_features.append(features)
                all_position_targets.append(pos_targets)
                all_speed_targets.append(speed_targets)
        
        if not all_features:
            print("No training data available")
            return False
        
        X = np.vstack(all_features)
        y_position = np.concatenate(all_position_targets)
        y_speed = np.concatenate(all_speed_targets)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_pos_train, y_pos_test = train_test_split(
            X_scaled, y_position, test_size=0.2, random_state=42
        )
        
        # Train position prediction model
        self.position_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.position_model.fit(X_train, y_pos_train)
        
        # Train speed prediction model
        X_train_sp, _, y_sp_train, _ = train_test_split(
            X_scaled, y_speed, test_size=0.2, random_state=42
        )
        self.laptime_model = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        self.laptime_model.fit(X_train_sp, y_sp_train)
        
        # Evaluate (silent - no print)
        pos_score = self.position_model.score(X_test, y_pos_test)
        
        self.is_trained = True
        return True

    def _get_cache_key(self, data_identifier: str) -> str:
        """Generate cache key from data identifier."""
        return hashlib.md5(data_identifier.encode()).hexdigest()

    def save_model(self, cache_key: str):
        """Save trained model to cache."""
        if not self.is_trained:
            return False
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            model_data = {
                'position_model': self.position_model,
                'laptime_model': self.laptime_model,
                'scaler': self.scaler,
                'is_trained': self.is_trained
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(model_data, f)
            return True
        except Exception as e:
            print(f"⚠️  無法儲存模型: {e}")
            return False

    def load_model(self, cache_key: str) -> bool:
        """Load trained model from cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if not cache_file.exists():
            return False
        
        try:
            with open(cache_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.position_model = model_data['position_model']
            self.laptime_model = model_data['laptime_model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            return True
        except Exception as e:
            print(f"⚠️  無法載入模型: {e}")
            return False

    def train(self, frames, drivers):
        """
        Train the prediction models using race data from all drivers.
        """
        if len(frames) < 500:
            print("Not enough frames for ML training (need at least 500)")
            return False

        all_features = []
        all_position_targets = []
        all_laptime_targets = []

        for driver in drivers:
            features, pos_targets, lap_targets = self.prepare_training_data(frames, driver)
            if len(features) > 0:
                all_features.append(features)
                all_position_targets.append(pos_targets)
                all_laptime_targets.append(lap_targets)

        if not all_features:
            print("No training data available")
            return False

        X = np.vstack(all_features)
        y_position = np.concatenate(all_position_targets)
        y_laptime = np.concatenate(all_laptime_targets)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_pos_train, y_pos_test = train_test_split(
            X_scaled, y_position, test_size=0.2, random_state=42
        )

        # Train position prediction model
        self.position_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.position_model.fit(X_train, y_pos_train)

        # Train lap time/speed prediction model
        X_train_lt, _, y_lt_train, _ = train_test_split(
            X_scaled, y_laptime, test_size=0.2, random_state=42
        )
        self.laptime_model = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        self.laptime_model.fit(X_train_lt, y_lt_train)

        # Evaluate (silent)
        pos_score = self.position_model.score(X_test, y_pos_test)

        self.is_trained = True
        return True

    def predict(self, frame, driver_code):
        """
        Predict future position and speed for a driver at a given frame.
        Returns prediction dict with confidence scores.
        """
        if not self.is_trained or driver_code not in frame['drivers']:
            return None

        driver_data = frame['drivers'][driver_code]

        feature_vector = np.array([[
            driver_data.get('position', 10),
            frame.get('lap', 1),
            driver_data.get('tyre', 1),
            driver_data.get('speed', 200),
            driver_data.get('gear', 5),
            1 if driver_data.get('drs', 0) in [10, 12, 14] else 0,
            driver_data.get('dist', 0),
            driver_data.get('rel_dist', 0),
        ]])

        X_scaled = self.scaler.transform(feature_vector)

        predicted_position = self.position_model.predict(X_scaled)[0]
        predicted_speed = self.laptime_model.predict(X_scaled)[0]

        # Calculate trend (improving = negative, declining = positive)
        current_position = driver_data.get('position', 10)
        position_trend = predicted_position - current_position

        # Calculate confidence based on prediction variance from the model
        # Using feature importances as a proxy for prediction reliability
        try:
            feature_importance_sum = sum(self.position_model.feature_importances_)
            confidence = min(0.95, max(0.5, feature_importance_sum / len(self.position_model.feature_importances_) * 2))
        except (AttributeError, ZeroDivisionError):
            confidence = 0.7

        prediction = {
            'driver': driver_code,
            'current_position': current_position,
            'predicted_position': round(predicted_position, 1),
            'position_change': round(position_trend, 1),
            'predicted_speed': round(predicted_speed, 1),
            'trend': 'improving' if position_trend < -0.5 else ('declining' if position_trend > 0.5 else 'stable'),
            'confidence': round(confidence, 2),
        }

        return prediction

    def predict_all_drivers(self, frame):
        """
        Generate predictions for all drivers in a frame.
        """
        predictions = {}
        for driver_code in frame['drivers'].keys():
            pred = self.predict(frame, driver_code)
            if pred:
                predictions[driver_code] = pred
        return predictions

    def get_race_insights(self, frame):
        """
        Generate high-level race insights based on current predictions.
        Returns list of insight strings for display.
        """
        if not self.is_trained:
            return ["ML model not trained yet - analyzing race data..."]

        predictions = self.predict_all_drivers(frame)
        if not predictions:
            return ["No predictions available"]

        insights = []

        # Find drivers with improving trends
        improving = [p for p in predictions.values() if p['trend'] == 'improving']
        declining = [p for p in predictions.values() if p['trend'] == 'declining']

        if improving:
            top_improver = min(improving, key=lambda x: x['position_change'])
            insights.append(
                f"🔥 {top_improver['driver']} showing strong pace, "
                f"predicted to gain {abs(top_improver['position_change']):.1f} positions"
            )

        if declining:
            top_decliner = max(declining, key=lambda x: x['position_change'])
            insights.append(
                f"⚠️ {top_decliner['driver']} losing ground, "
                f"may drop {abs(top_decliner['position_change']):.1f} positions"
            )

        # Battle predictions
        sorted_by_pos = sorted(predictions.values(), key=lambda x: x['current_position'])
        for i in range(min(3, len(sorted_by_pos) - 1)):
            curr = sorted_by_pos[i]
            next_driver = sorted_by_pos[i + 1]
            if abs(curr['predicted_position'] - next_driver['predicted_position']) < 0.5:
                insights.append(
                    f"⚔️ Battle brewing: P{int(curr['current_position'])} "
                    f"{curr['driver']} vs P{int(next_driver['current_position'])} {next_driver['driver']}"
                )

        if not insights:
            insights.append("📊 Race positions stable, no major changes predicted")

        return insights[:4]  # Limit to 4 insights

    def analyze_pit_strategy(self, frames, driver_code, current_lap):
        """
        Analyze optimal pit stop strategy based on tyre degradation patterns.
        """
        if not self.is_trained:
            return None

        # Simple heuristic-based pit window analysis
        tyre_stints = {}
        current_tyre = None
        stint_start = 0

        for i, frame in enumerate(frames):
            if driver_code not in frame['drivers']:
                continue
            driver = frame['drivers'][driver_code]
            tyre = driver.get('tyre', 1)

            if tyre != current_tyre:
                if current_tyre is not None:
                    stint_length = i - stint_start
                    if current_tyre not in tyre_stints:
                        tyre_stints[current_tyre] = []
                    tyre_stints[current_tyre].append(stint_length)
                current_tyre = tyre
                stint_start = i

        # Estimate optimal pit window based on tyre compound
        compound_names = {0: "SOFT", 1: "MEDIUM", 2: "HARD", 3: "INTER", 4: "WET"}
        current_compound = compound_names.get(current_tyre, "MEDIUM")

        # Estimated tyre life in laps
        tyre_life = {
            "SOFT": (15, 22),
            "MEDIUM": (25, 35),
            "HARD": (35, 50),
            "INTER": (20, 40),
            "WET": (20, 40),
        }

        life = tyre_life.get(current_compound, (20, 30))

        return {
            'current_tyre': current_compound,
            'estimated_pit_window': (current_lap + life[0], current_lap + life[1]),
            'recommendation': f"Consider pitting between lap {current_lap + life[0]} and {current_lap + life[1]}"
        }


class PreRacePredictor:
    """
    Enhanced predictor for future race outcomes.
    Uses historical season data to predict race results before they happen.
    """

    # Team strength mapping (lower = stronger)
    TEAM_STRENGTH = {
        'Red Bull': 1.0,
        'McLaren': 1.5,
        'Ferrari': 2.0,
        'Mercedes': 2.5,
        'Aston Martin': 4.5,
        'Williams': 6.0,
        'RB': 5.5,
        'Alpine': 6.5,
        'Haas': 7.0,
        'Sauber': 7.5,
    }

    def __init__(self, cache_dir: str = "cache/ml_models"):
        """Initialize the pre-race predictor."""
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_years = []
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def prepare_historical_data(self, year: int):
        """
        Fetch historical race results to train the model.

        Args:
            year: Year to fetch data from

        Returns:
            Tuple of (X_train, y_train) numpy arrays
        """
        import fastf1
        import pandas as pd

        X_train = []
        y_train = []

        try:
            schedule = fastf1.get_event_schedule(year)
            completed_races = schedule[schedule['EventDate'] < pd.Timestamp.now()]

            # Reduced verbosity
            for idx, (_, race_event) in enumerate(completed_races.iterrows(), 1):
                try:
                    # Show progress every 5 races
                    if idx % 5 == 1 or idx == len(completed_races):
                        print(f"   處理進度: {idx}/{len(completed_races)} 場比賽...")
                    
                    session = fastf1.get_session(year, race_event['RoundNumber'], 'R')
                    session.load(telemetry=False, weather=False, messages=False)

                    results = session.results

                    team_strength = results.groupby('TeamName')['Position'].mean().to_dict()

                    for driver in results.index:
                        d_data = results.loc[driver]

                        valid_positions = ['R', 'F'] + [str(i) for i in range(1, 21)]
                        if d_data['ClassifiedPosition'] not in valid_positions:
                            continue

                        grid_pos = d_data['GridPosition']
                        team_score = team_strength.get(d_data['TeamName'], 10)
                        points = d_data['Points']

                        X_train.append([grid_pos, team_score, points])
                        y_train.append(d_data['Position'])

                except Exception:
                    # Silently skip failed races
                    continue

        except Exception:
            # Silently fail if can't get schedule
            pass

        return np.array(X_train), np.array(y_train)

    def _get_cache_key(self, years: list) -> str:
        """Generate cache key from training years."""
        years_str = "_".join(str(y) for y in sorted(years))
        return f"prerace_model_{years_str}"

    def save_model(self, years: list):
        """Save trained model to cache."""
        if not self.is_trained:
            return False
        
        cache_key = self._get_cache_key(years)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained,
                'training_years': self.training_years
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"✅ 模型已儲存至快取")
            return True
        except Exception as e:
            print(f"⚠️  無法儲存模型: {e}")
            return False

    def load_model(self, years: list) -> bool:
        """Load trained model from cache."""
        cache_key = self._get_cache_key(years)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return False
        
        try:
            with open(cache_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            self.training_years = model_data['training_years']
            print(f"✅ 從快取載入模型 (訓練年份: {', '.join(map(str, years))})")
            return True
        except Exception as e:
            print(f"⚠️  無法載入模型: {e}")
            return False

    def train_on_historical_data(self, years: list = None):
        """
        Train the model on multiple years of historical data.

        Args:
            years: List of years to train on (default: [2023, 2024])

        Returns:
            True if training succeeded, False otherwise
        """
        if years is None:
            years = [2023, 2024]

        # Try loading from cache first
        if self.load_model(years):
            return True

        all_X = []
        all_y = []

        for year in years:
            print(f"   載入 {year} 賽季數據...")
            X, y = self.prepare_historical_data(year)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
                print(f"   ✓ {year}: {len(X)} 個數據點")

        if not all_X:
            print("❌ 訓練失敗：沒有足夠的歷史數據")
            return False

        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_combined)

        # Train model
        print(f"   正在訓練模型...")
        self.model.fit(X_scaled, y_combined)
        self.is_trained = True
        self.training_years = years

        # Save to cache
        self.save_model(years)

        return True

    def train_for_season(self, year: int = 2023):
        """
        Train the model based on past races of a single year.
        Backward compatible method.

        Args:
            year: Year to train on

        Returns:
            True if training succeeded
        """
        return self.train_on_historical_data([year])

    def predict_race_result(self, qualifying_positions: list, track_type: str = "permanent") -> list:
        """
        Predict race results based on qualifying positions.

        Args:
            qualifying_positions: List of dicts with driver, grid, team, points
            track_type: Type of track ("permanent", "street")

        Returns:
            List of predictions sorted by predicted finish
        """
        if not self.is_trained:
            return []

        predictions = []

        for driver in qualifying_positions:
            team_score = self.TEAM_STRENGTH.get(driver['team'], 10.0)

            features = np.array([[
                driver['grid'],
                team_score,
                driver.get('points', 0)
            ]])

            X_scaled = self.scaler.transform(features)
            predicted_pos = self.model.predict(X_scaled)[0]

            # Calculate confidence based on grid position and team strength
            confidence = self._calculate_confidence(driver['grid'], team_score)

            predictions.append({
                'driver': driver['driver'],
                'name': driver.get('name', driver['driver']),
                'team': driver['team'],
                'grid': driver['grid'],
                'predicted_finish': round(predicted_pos, 1),
                'confidence': round(confidence, 2),
            })

        predictions.sort(key=lambda x: x['predicted_finish'])

        # Assign final positions
        for i, pred in enumerate(predictions):
            pred['position'] = i + 1

        return predictions

    def _calculate_confidence(self, grid: int, team_strength: float) -> float:
        """Calculate prediction confidence."""
        # Front runners with strong teams have higher confidence
        position_factor = max(0.5, 1 - grid * 0.03)
        team_factor = max(0.5, 1 - (team_strength - 1) * 0.08)
        return min(0.95, position_factor * team_factor)

    def predict_next_race(self, qualifying_results: list) -> list:
        """
        Predict the outcome of the next race based on qualifying results.
        Backward compatible method.

        Args:
            qualifying_results: List of dicts with driver, grid, team, points

        Returns:
            Sorted list of predictions
        """
        return self.predict_race_result(qualifying_results)

    def predict_lap_by_lap(self, qualifying_positions: list, total_laps: int) -> list:
        """
        Predict position changes for each lap of the race.

        Args:
            qualifying_positions: List of driver qualifying data
            total_laps: Total number of laps in the race

        Returns:
            List of lap-by-lap predictions (frames)
        """
        import random

        if not self.is_trained:
            return []

        # Initialize positions from qualifying
        current_positions = {}
        for quali in qualifying_positions:
            current_positions[quali['driver']] = {
                'position': quali['grid'],
                'team': quali['team'],
                'pace': 1.0 - (self.TEAM_STRENGTH.get(quali['team'], 10) - 1) * 0.02,
                'tyre_age': 0,
            }

        lap_predictions = []

        for lap in range(1, total_laps + 1):
            # Update tyre degradation
            for driver, state in current_positions.items():
                state['tyre_age'] += 1
                degradation = state['tyre_age'] * 0.001
                state['current_pace'] = state['pace'] - degradation

            # Simulate potential position changes
            sorted_drivers = sorted(
                current_positions.items(),
                key=lambda x: x[1]['position']
            )

            # Check for overtakes
            for i in range(len(sorted_drivers) - 1):
                ahead = sorted_drivers[i]
                behind = sorted_drivers[i + 1]

                pace_diff = behind[1]['current_pace'] - ahead[1]['current_pace']
                if pace_diff > 0.01 and random.random() < 0.3:
                    # Swap positions
                    ahead_pos = current_positions[ahead[0]]['position']
                    behind_pos = current_positions[behind[0]]['position']
                    current_positions[ahead[0]]['position'] = behind_pos
                    current_positions[behind[0]]['position'] = ahead_pos

            # Record lap state
            lap_predictions.append({
                'lap': lap,
                'positions': {
                    driver: {
                        'position': state['position'],
                        'trend': 'stable',
                        'confidence': 0.8,
                    }
                    for driver, state in current_positions.items()
                }
            })

        return lap_predictions

    def get_training_info(self) -> dict:
        """Get information about the trained model."""
        return {
            'is_trained': self.is_trained,
            'training_years': self.training_years,
            'model_type': 'RandomForestRegressor',
        }
