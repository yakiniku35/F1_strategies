#!/usr/bin/env python3
"""
Demo script for Enhanced ML Prediction Model.

This script demonstrates the new features:
1. Ensemble learning (RF + GradientBoosting + XGBoost)
2. Driver form tracking
3. Confidence intervals for predictions
4. Advanced feature engineering
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml_enhanced import EnhancedRacePredictor, DriverFormTracker


def demo_driver_form_tracking():
    """Demonstrate driver form tracking."""
    print("=" * 70)
    print("Demo 1: Driver Form Tracking")
    print("=" * 70)
    
    tracker = DriverFormTracker(lookback_races=5)
    
    # Simulate Verstappen's recent performance
    races = [
        ('VER', 'Red Bull Racing', 1, 1, 25),  # P1 from P1
        ('VER', 'Red Bull Racing', 2, 1, 18),  # P2 from P1
        ('VER', 'Red Bull Racing', 1, 2, 25),  # P1 from P2
        ('VER', 'Red Bull Racing', 1, 1, 25),  # P1 from P1
        ('VER', 'Red Bull Racing', 3, 1, 15),  # P3 from P1
    ]
    
    for driver, team, position, quali, points in races:
        tracker.update(driver, team, position, quali, points)
    
    form = tracker.get_driver_form('VER')
    
    print("\n📊 Max Verstappen Recent Form:")
    print(f"   Average Position: P{form['avg_position']:.1f}")
    print(f"   Average Gain: {form['avg_gain']:.2f} positions")
    print(f"   Consistency: {form['consistency']:.2f}")
    print(f"   Momentum: {form['momentum']:.2f}")
    print(f"   Points Rate: {form['points_rate']:.1f} pts/race")
    
    # Simulate Norris improving form
    races_nor = [
        ('NOR', 'McLaren', 5, 6, 10),
        ('NOR', 'McLaren', 3, 5, 15),
        ('NOR', 'McLaren', 2, 3, 18),
        ('NOR', 'McLaren', 1, 2, 25),
        ('NOR', 'McLaren', 1, 1, 25),
    ]
    
    for driver, team, position, quali, points in races_nor:
        tracker.update(driver, team, position, quali, points)
    
    form_nor = tracker.get_driver_form('NOR')
    
    print("\n📊 Lando Norris Recent Form (Improving):")
    print(f"   Average Position: P{form_nor['avg_position']:.1f}")
    print(f"   Average Gain: {form_nor['avg_gain']:.2f} positions")
    print(f"   Consistency: {form_nor['consistency']:.2f}")
    print(f"   Momentum: {form_nor['momentum']:.2f} (⬆️ Improving!)")
    print(f"   Points Rate: {form_nor['points_rate']:.1f} pts/race")


def demo_enhanced_prediction():
    """Demonstrate enhanced prediction with confidence intervals."""
    print("\n" + "=" * 70)
    print("Demo 2: Enhanced Race Prediction with Confidence Intervals")
    print("=" * 70)
    
    predictor = EnhancedRacePredictor()
    
    print("\n⏳ Training model on historical data (2022-2024)...")
    print("   This may take a few minutes on first run...")
    
    success = predictor.train_from_historical_data(
        years=[2023, 2024],  # Use 2023-2024 for faster demo
        verbose=True
    )
    
    if not success:
        print("\n⚠️  Training failed. Using simulated predictions instead.")
        demo_simulated_predictions()
        return
    
    # Get model info
    info = predictor.get_model_info()
    print(f"\n✅ Model trained successfully!")
    print(f"   Models used: {', '.join(info['models'])}")
    print(f"   Position R²: {info['scores']['position']['test_r2']:.3f}")
    print(f"   Laptime R²: {info['scores']['laptime']['test_r2']:.3f}")
    
    # Example: Predict Monaco 2025
    print("\n" + "-" * 70)
    print("🔮 Predicting Monaco 2025")
    print("-" * 70)
    
    qualifying_data = [
        {'driver': 'Max Verstappen', 'code': 'VER', 'grid': 1, 'team': 'Red Bull Racing'},
        {'driver': 'Charles Leclerc', 'code': 'LEC', 'grid': 2, 'team': 'Ferrari'},
        {'driver': 'Lando Norris', 'code': 'NOR', 'grid': 3, 'team': 'McLaren'},
        {'driver': 'Oscar Piastri', 'code': 'PIA', 'grid': 4, 'team': 'McLaren'},
        {'driver': 'Carlos Sainz', 'code': 'SAI', 'grid': 5, 'team': 'Ferrari'},
        {'driver': 'Lewis Hamilton', 'code': 'HAM', 'grid': 6, 'team': 'Mercedes'},
        {'driver': 'George Russell', 'code': 'RUS', 'grid': 7, 'team': 'Mercedes'},
        {'driver': 'Fernando Alonso', 'code': 'ALO', 'grid': 8, 'team': 'Aston Martin'},
    ]
    
    predictions = predictor.predict_race_result(qualifying_data, track_name='Monaco')
    
    print("\n📊 Predicted Race Results:")
    print(f"{'Pos':<5} {'Driver':<20} {'Grid':<6} {'Predicted':<12} {'Confidence':<12}")
    print("-" * 70)
    
    for i, pred in enumerate(predictions[:10], 1):
        conf_range = f"P{pred['confidence_lower']:.0f}-P{pred['confidence_upper']:.0f}"
        print(f"{i:<5} {pred['driver']:<20} P{pred['grid']:<5} "
              f"P{pred['predicted_position']:<11.1f} {conf_range:<12} "
              f"({pred['confidence']*100:.0f}%)")
    
    # Highlight interesting predictions
    print("\n💡 Key Insights:")
    for pred in predictions[:10]:
        gain = pred['grid'] - pred['predicted_position']
        if gain > 2:
            print(f"   🔥 {pred['driver']} expected to gain ~{gain:.0f} positions!")
        elif gain < -2:
            print(f"   ⚠️  {pred['driver']} may lose ~{abs(gain):.0f} positions")


def demo_simulated_predictions():
    """Show simulated predictions if training fails."""
    print("\n📊 Simulated Predictions (Demo Mode):")
    print(f"{'Pos':<5} {'Driver':<20} {'Grid':<6} {'Predicted':<12} {'Confidence'}")
    print("-" * 70)
    
    simulated = [
        (1, 'Max Verstappen', 1, 1.2, 0.92),
        (2, 'Charles Leclerc', 2, 2.1, 0.88),
        (3, 'Lando Norris', 3, 2.8, 0.85),
        (4, 'Carlos Sainz', 5, 4.3, 0.78),
        (5, 'Oscar Piastri', 4, 4.9, 0.82),
    ]
    
    for pos, driver, grid, pred, conf in simulated:
        print(f"{pos:<5} {driver:<20} P{grid:<5} P{pred:<11.1f} {conf*100:.0f}%")


def demo_track_type_analysis():
    """Demonstrate track-specific feature engineering."""
    print("\n" + "=" * 70)
    print("Demo 3: Track-Type Specific Analysis")
    print("=" * 70)
    
    predictor = EnhancedRacePredictor()
    
    tracks = [
        ('Monaco', 'street'),
        ('Monza', 'high_speed'),
        ('Suzuka', 'technical'),
        ('Silverstone', 'high_speed'),
    ]
    
    print("\n🏁 Track Type Classification:")
    for track, expected_type in tracks:
        features = predictor._get_track_type_features(track)
        detected_type = [k.replace('is_', '') for k, v in features.items() if v == 1]
        status = "✓" if expected_type in detected_type else "✗"
        print(f"   {status} {track:<15} → {', '.join(detected_type) or 'mixed'}")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("🏎️  F1 Enhanced ML Prediction Model - Demo")
    print("=" * 70)
    
    try:
        # Demo 1: Driver form tracking
        demo_driver_form_tracking()
        
        # Demo 2: Enhanced predictions
        demo_enhanced_prediction()
        
        # Demo 3: Track type analysis
        demo_track_type_analysis()
        
        print("\n" + "=" * 70)
        print("✅ Demo Complete!")
        print("\nTo use enhanced predictions in your race simulator:")
        print("   from src.ml_enhanced import EnhancedRacePredictor")
        print("   predictor = EnhancedRacePredictor()")
        print("   predictor.train_from_historical_data()")
        print("   predictions = predictor.predict_race_result(qualifying_data, 'Monaco')")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
