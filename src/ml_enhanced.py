"""
Enhanced Machine Learning module for F1 race predictions.
Implements advanced ML techniques including:
- Ensemble models (Random Forest + XGBoost + Gradient Boosting)
- Feature engineering (weather, track type, driver form)
- Confidence intervals and uncertainty quantification
- Driver performance tracking over time
"""

import numpy as np
import pickle
import warnings
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from collections import defaultdict
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")


class DriverFormTracker:
    """Tracks driver performance trends over recent races."""
    
    def __init__(self, lookback_races: int = 5):
        """
        Initialize form tracker.
        
        Args:
            lookback_races: Number of recent races to consider for form
        """
        self.lookback_races = lookback_races
        self.driver_history = defaultdict(list)
        self.team_history = defaultdict(list)
        
    def update(self, driver_code: str, team: str, position: int, 
               qualifying_pos: int, points: float):
        """
        Update driver and team performance history.
        
        Args:
            driver_code: Driver code (e.g., "VER", "HAM")
            team: Team name
            position: Finish position
            qualifying_pos: Qualifying position
            points: Points scored
        """
        race_data = {
            'position': position,
            'qualifying': qualifying_pos,
            'points': points,
            'gain': qualifying_pos - position  # Positive = gained positions
        }
        
        self.driver_history[driver_code].append(race_data)
        self.team_history[team].append(race_data)
        
        # Keep only recent races
        if len(self.driver_history[driver_code]) > self.lookback_races:
            self.driver_history[driver_code].pop(0)
        if len(self.team_history[team]) > self.lookback_races:
            self.team_history[team].pop(0)
    
    def get_driver_form(self, driver_code: str) -> dict:
        """
        Calculate driver form metrics.
        
        Returns:
            Dict with form metrics: avg_position, avg_gain, consistency, momentum
        """
        history = self.driver_history.get(driver_code, [])
        if not history:
            return {
                'avg_position': 10.0,
                'avg_gain': 0.0,
                'consistency': 0.5,
                'momentum': 0.0,
                'points_rate': 0.0
            }
        
        positions = [r['position'] for r in history]
        gains = [r['gain'] for r in history]
        points = [r['points'] for r in history]
        
        # Average finish position (lower is better)
        avg_position = np.mean(positions)
        
        # Average position gain from qualifying (positive = improving)
        avg_gain = np.mean(gains)
        
        # Consistency (lower std dev = more consistent)
        consistency = 1.0 / (1.0 + np.std(positions))
        
        # Momentum: recent trend (weighted average of recent gains)
        if len(gains) >= 3:
            weights = np.array([0.2, 0.3, 0.5])  # Recent races weighted more
            momentum = np.average(gains[-3:], weights=weights)
        else:
            momentum = avg_gain
        
        # Points scoring rate
        points_rate = np.mean(points)
        
        return {
            'avg_position': avg_position,
            'avg_gain': avg_gain,
            'consistency': consistency,
            'momentum': momentum,
            'points_rate': points_rate
        }
    
    def get_team_form(self, team: str) -> dict:
        """Calculate team form metrics."""
        history = self.team_history.get(team, [])
        if not history:
            return {
                'avg_position': 10.0,
                'reliability': 0.8
            }
        
        positions = [r['position'] for r in history]
        
        return {
            'avg_position': np.mean(positions),
            'reliability': len([p for p in positions if p <= 20]) / len(positions)
        }


class EnhancedRacePredictor:
    """
    Enhanced race predictor using ensemble methods and advanced features.
    Combines multiple ML models for improved accuracy.
    """
    
    # Team performance ratings (updated for 2024/2025)
    TEAM_RATINGS = {
        'Red Bull Racing': 0.95,
        'Red Bull': 0.95,
        'McLaren': 0.90,
        'Ferrari': 0.88,
        'Mercedes': 0.85,
        'Aston Martin': 0.72,
        'Alpine': 0.68,
        'Williams': 0.65,
        'RB': 0.70,
        'Haas': 0.63,
        'Sauber': 0.60,
        'Kick Sauber': 0.60,
    }
    
    # Track type characteristics
    TRACK_TYPES = {
        'high_speed': ['Monza', 'Spa', 'Silverstone', 'Jeddah'],
        'street': ['Monaco', 'Singapore', 'Baku', 'Las Vegas', 'Miami'],
        'technical': ['Barcelona', 'Suzuka', 'Hungary', 'Zandvoort'],
        'mixed': ['Austin', 'Bahrain', 'Abu Dhabi', 'Saudi Arabia']
    }
    
    def __init__(self, cache_dir: str = "cache/ml_models"):
        """Initialize enhanced predictor."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.position_model = None
        self.laptime_model = None
        self.ensemble_model = None
        
        # Scalers
        self.position_scaler = StandardScaler()
        self.laptime_scaler = StandardScaler()
        
        # Form tracker
        self.form_tracker = DriverFormTracker(lookback_races=5)
        
        # Training state
        self.is_trained = False
        self.feature_names = []
        self.model_scores = {}
        
    def _get_track_type_features(self, track_name: str) -> dict:
        """
        Get track type one-hot encoding.
        
        Returns:
            Dict with track type indicators
        """
        features = {
            'is_high_speed': 0,
            'is_street': 0,
            'is_technical': 0,
            'is_mixed': 0
        }
        
        for track_type, tracks in self.TRACK_TYPES.items():
            if any(track.lower() in track_name.lower() for track in tracks):
                features[f'is_{track_type}'] = 1
                break
        
        return features
    
    def _engineer_features(self, base_features: np.ndarray, 
                          driver_codes: list = None,
                          track_name: str = None) -> np.ndarray:
        """
        Engineer advanced features from base features.
        
        Args:
            base_features: Array of [position, lap, tyre, speed, gear, drs, dist, rel_dist]
            driver_codes: List of driver codes for form lookup
            track_name: Track name for track-specific features
            
        Returns:
            Enhanced feature array
        """
        n_samples = base_features.shape[0]
        enhanced_features = []
        
        for i in range(n_samples):
            position, lap, tyre, speed, gear, drs, dist, rel_dist = base_features[i]
            
            # Base features
            features = list(base_features[i])
            
            # Derived features
            features.extend([
                position / 20.0,  # Normalized position
                lap / 60.0,  # Normalized lap (assume ~60 lap race)
                speed / 350.0,  # Normalized speed
                gear / 8.0,  # Normalized gear
                1 if position <= 3 else 0,  # Is in podium positions
                1 if position <= 10 else 0,  # Is in points positions
                tyre * 0.1,  # Tyre compound factor
                abs(rel_dist) / 1000.0,  # Normalized relative distance
            ])
            
            # Driver form features (if available)
            if driver_codes and i < len(driver_codes):
                driver_form = self.form_tracker.get_driver_form(driver_codes[i])
                features.extend([
                    driver_form['avg_position'] / 20.0,
                    driver_form['avg_gain'] / 10.0,
                    driver_form['consistency'],
                    driver_form['momentum'] / 5.0,
                    driver_form['points_rate'] / 25.0,
                ])
            else:
                features.extend([0.5, 0.0, 0.5, 0.0, 0.0])  # Default form
            
            # Track type features
            if track_name:
                track_features = self._get_track_type_features(track_name)
                features.extend(list(track_features.values()))
            else:
                features.extend([0, 0, 0, 0])  # No track info
            
            enhanced_features.append(features)
        
        return np.array(enhanced_features)
    
    def _build_ensemble_model(self):
        """Build ensemble model combining multiple estimators."""
        estimators = []
        
        # Random Forest: Good for non-linear relationships
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        estimators.append(('rf', rf))
        
        # Gradient Boosting: Good for capturing complex patterns
        gb = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        estimators.append(('gb', gb))
        
        # XGBoost: State-of-art gradient boosting (if available)
        if XGBOOST_AVAILABLE:
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            estimators.append(('xgb', xgb_model))
        
        # Create voting regressor (averages predictions)
        ensemble = VotingRegressor(estimators=estimators)
        
        return ensemble
    
    def train_position_predictor(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train position prediction model with cross-validation.
        
        Returns:
            Dict with training metrics
        """
        # Scale features
        X_scaled = self.position_scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Build and train ensemble
        self.position_model = self._build_ensemble_model()
        self.position_model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.position_model.score(X_train, y_train)
        test_score = self.position_model.score(X_test, y_test)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.position_model, X_scaled, y, cv=5, scoring='r2'
        )
        
        metrics = {
            'train_r2': train_score,
            'test_r2': test_score,
            'cv_mean_r2': np.mean(cv_scores),
            'cv_std_r2': np.std(cv_scores)
        }
        
        return metrics
    
    def train_laptime_predictor(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train lap time/speed prediction model."""
        X_scaled = self.laptime_scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        self.laptime_model = self._build_ensemble_model()
        self.laptime_model.fit(X_train, y_train)
        
        train_score = self.laptime_model.score(X_train, y_train)
        test_score = self.laptime_model.score(X_test, y_test)
        
        cv_scores = cross_val_score(
            self.laptime_model, X_scaled, y, cv=5, scoring='r2'
        )
        
        metrics = {
            'train_r2': train_score,
            'test_r2': test_score,
            'cv_mean_r2': np.mean(cv_scores),
            'cv_std_r2': np.std(cv_scores)
        }
        
        return metrics
    
    def train_from_historical_data(self, years: list = None, 
                                   verbose: bool = True) -> bool:
        """
        Train models on historical F1 data.
        
        Args:
            years: List of years to train on (default: [2022, 2023, 2024])
            verbose: Print training progress
            
        Returns:
            True if training succeeded
        """
        if years is None:
            years = [2022, 2023, 2024]
        
        # Try loading cached model
        cache_key = f"enhanced_model_{'_'.join(map(str, years))}"
        if self._load_from_cache(cache_key):
            if verbose:
                print(f"✅ Loaded cached model (trained on {years})")
            return True
        
        if verbose:
            print(f"🔄 Training enhanced model on {years}...")
        
        try:
            import fastf1
            import pandas as pd
            
            all_features = []
            all_position_targets = []
            all_laptime_targets = []
            
            for year in years:
                if verbose:
                    print(f"   📅 Processing {year} season...")
                
                try:
                    schedule = fastf1.get_event_schedule(year)
                    completed_races = schedule[schedule['EventDate'] < pd.Timestamp.now()]
                    
                    for idx, (_, race_event) in enumerate(completed_races.iterrows(), 1):
                        try:
                            if verbose and idx % 5 == 0:
                                print(f"      Race {idx}/{len(completed_races)}...")
                            
                            session = fastf1.get_session(year, race_event['RoundNumber'], 'R')
                            session.load(telemetry=False, weather=False, messages=False)
                            
                            results = session.results
                            track_name = race_event.get('EventName', 'Unknown')
                            
                            # Extract features for each driver
                            for driver_idx, driver in enumerate(results.index):
                                d_data = results.loc[driver]
                                
                                # Skip if didn't finish
                                if pd.isna(d_data['Position']) or d_data['Position'] == 0:
                                    continue
                                
                                # Base features
                                base_features = np.array([[
                                    d_data.get('GridPosition', 10),
                                    35,  # Mid-race lap proxy
                                    1,  # Medium tyre proxy
                                    250,  # Average speed proxy
                                    6,  # Average gear
                                    0,  # DRS off
                                    5000,  # Distance proxy
                                    0,  # Relative distance
                                ]])
                                
                                # Engineer features
                                driver_code = d_data.get('Abbreviation', 'UNK')
                                enhanced = self._engineer_features(
                                    base_features,
                                    driver_codes=[driver_code],
                                    track_name=track_name
                                )
                                
                                all_features.append(enhanced[0])
                                all_position_targets.append(d_data['Position'])
                                
                                # Lap time target (use average if available)
                                if 'Time' in d_data and not pd.isna(d_data['Time']):
                                    total_seconds = d_data['Time'].total_seconds()
                                    avg_laptime = total_seconds / race_event.get('RoundNumber', 50)
                                    all_laptime_targets.append(avg_laptime)
                                else:
                                    all_laptime_targets.append(90.0)  # Default ~90s lap
                                
                                # Update form tracker
                                team_name = d_data.get('TeamName', 'Unknown')
                                self.form_tracker.update(
                                    driver_code=driver_code,
                                    team=team_name,
                                    position=int(d_data['Position']),
                                    qualifying_pos=int(d_data.get('GridPosition', 10)),
                                    points=float(d_data.get('Points', 0))
                                )
                        
                        except Exception as e:
                            if verbose:
                                print(f"      ⚠️  Skipped race {idx}: {str(e)[:50]}")
                            continue
                
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️  Failed to load {year}: {str(e)[:50]}")
                    continue
            
            if not all_features:
                print("❌ No training data available")
                return False
            
            X = np.array(all_features)
            y_position = np.array(all_position_targets)
            y_laptime = np.array(all_laptime_targets)
            
            if verbose:
                print(f"   📊 Training on {len(X)} samples...")
            
            # Train position predictor
            pos_metrics = self.train_position_predictor(X, y_position)
            self.model_scores['position'] = pos_metrics
            
            if verbose:
                print(f"   ✓ Position model: R² = {pos_metrics['test_r2']:.3f} "
                      f"(CV: {pos_metrics['cv_mean_r2']:.3f} ± {pos_metrics['cv_std_r2']:.3f})")
            
            # Train laptime predictor
            lap_metrics = self.train_laptime_predictor(X, y_laptime)
            self.model_scores['laptime'] = lap_metrics
            
            if verbose:
                print(f"   ✓ Laptime model: R² = {lap_metrics['test_r2']:.3f} "
                      f"(CV: {lap_metrics['cv_mean_r2']:.3f} ± {lap_metrics['cv_std_r2']:.3f})")
            
            self.is_trained = True
            
            # Save to cache
            self._save_to_cache(cache_key)
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"❌ Training failed: {e}")
            return False
    
    def predict_with_confidence(self, features: np.ndarray, 
                               prediction_type: str = 'position') -> dict:
        """
        Make prediction with confidence interval.
        
        Args:
            features: Feature array
            prediction_type: 'position' or 'laptime'
            
        Returns:
            Dict with prediction, confidence_lower, confidence_upper
        """
        if not self.is_trained:
            return None
        
        # Select model and scaler
        if prediction_type == 'position':
            model = self.position_model
            scaler = self.position_scaler
        else:
            model = self.laptime_model
            scaler = self.laptime_scaler
        
        # Scale features
        X_scaled = scaler.transform(features)
        
        # Get predictions from each estimator
        predictions = []
        for name, estimator in model.named_estimators_.items():
            pred = estimator.predict(X_scaled)
            predictions.append(pred[0])
        
        # Calculate statistics
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        
        # 95% confidence interval (±1.96 std dev)
        confidence_lower = pred_mean - 1.96 * pred_std
        confidence_upper = pred_mean + 1.96 * pred_std
        
        return {
            'prediction': pred_mean,
            'confidence_lower': max(1, confidence_lower) if prediction_type == 'position' else confidence_lower,
            'confidence_upper': min(20, confidence_upper) if prediction_type == 'position' else confidence_upper,
            'std_dev': pred_std,
            'confidence_score': min(1.0, 1.0 / (1.0 + pred_std))
        }
    
    def predict_race_result(self, qualifying_data: list, track_name: str = None) -> list:
        """
        Predict race results from qualifying positions.
        
        Args:
            qualifying_data: List of dicts with driver, grid, team, code
            track_name: Track name for track-specific features
            
        Returns:
            List of predictions with confidence intervals
        """
        if not self.is_trained:
            return []
        
        predictions = []
        
        for quali in qualifying_data:
            # Build base features
            base_features = np.array([[
                quali['grid'],
                35,  # Mid-race proxy
                1,  # Medium tyre
                250,  # Average speed
                6,  # Average gear
                0,  # DRS off
                5000,  # Distance
                0,  # Rel distance
            ]])
            
            # Engineer features
            driver_code = quali.get('code', 'UNK')
            enhanced = self._engineer_features(
                base_features,
                driver_codes=[driver_code],
                track_name=track_name
            )
            
            # Predict position with confidence
            pos_result = self.predict_with_confidence(enhanced, 'position')
            
            if pos_result:
                predictions.append({
                    'driver': quali['driver'],
                    'code': driver_code,
                    'team': quali['team'],
                    'grid': quali['grid'],
                    'predicted_position': round(pos_result['prediction'], 1),
                    'confidence_lower': round(pos_result['confidence_lower'], 1),
                    'confidence_upper': round(pos_result['confidence_upper'], 1),
                    'confidence': round(pos_result['confidence_score'], 2),
                })
        
        # Sort by predicted position
        predictions.sort(key=lambda x: x['predicted_position'])
        
        return predictions
    
    def _save_to_cache(self, cache_key: str) -> bool:
        """Save model to cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            data = {
                'position_model': self.position_model,
                'laptime_model': self.laptime_model,
                'position_scaler': self.position_scaler,
                'laptime_scaler': self.laptime_scaler,
                'form_tracker': self.form_tracker,
                'model_scores': self.model_scores,
                'is_trained': self.is_trained
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"⚠️  Cache save failed: {e}")
            return False
    
    def _load_from_cache(self, cache_key: str) -> bool:
        """Load model from cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if not cache_file.exists():
            return False
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            self.position_model = data['position_model']
            self.laptime_model = data['laptime_model']
            self.position_scaler = data['position_scaler']
            self.laptime_scaler = data['laptime_scaler']
            self.form_tracker = data['form_tracker']
            self.model_scores = data['model_scores']
            self.is_trained = data['is_trained']
            
            return True
        except Exception:
            return False
    
    def get_model_info(self) -> dict:
        """Get information about trained models."""
        if not self.is_trained:
            return {'is_trained': False}
        
        info = {
            'is_trained': True,
            'models': ['RandomForest', 'GradientBoosting'],
            'scores': self.model_scores
        }
        
        if XGBOOST_AVAILABLE:
            info['models'].append('XGBoost')
        
        return info