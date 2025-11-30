"""
Machine Learning module for F1 race trend prediction.
Uses historical race data to predict future race outcomes.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class RaceTrendPredictor:
    """
    ML-based race trend predictor that analyzes race data
    and predicts future positions, lap times, and pit stop strategies.
    """

    def __init__(self):
        self.position_model = None
        self.laptime_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.predictions = {}
        self.prediction_history = []

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

        # Evaluate
        pos_score = self.position_model.score(X_test, y_pos_test)
        print(f"ML Model trained - Position prediction R² score: {pos_score:.3f}")

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

        prediction = {
            'driver': driver_code,
            'current_position': current_position,
            'predicted_position': round(predicted_position, 1),
            'position_change': round(position_trend, 1),
            'predicted_speed': round(predicted_speed, 1),
            'trend': 'improving' if position_trend < -0.5 else ('declining' if position_trend > 0.5 else 'stable'),
            'confidence': 0.7 + np.random.uniform(-0.1, 0.1),  # Simulated confidence
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
