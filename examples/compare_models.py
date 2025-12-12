#!/usr/bin/env python3
"""
Model Comparison: Standard vs Enhanced ML
Shows the improvements of the enhanced model over the standard model.
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml_predictor import PreRacePredictor
from src.ml_enhanced import EnhancedRacePredictor


def compare_predictions():
    """Compare predictions between standard and enhanced models."""
    print("=" * 80)
    print("ML Model Comparison: Standard vs Enhanced")
    print("=" * 80)
    
    # Qualifying data for testing
    qualifying_data = [
        {'driver': 'Max Verstappen', 'code': 'VER', 'grid': 1, 'team': 'Red Bull', 'points': 575},
        {'driver': 'Lando Norris', 'code': 'NOR', 'grid': 2, 'team': 'McLaren', 'points': 356},
        {'driver': 'Charles Leclerc', 'code': 'LEC', 'grid': 3, 'team': 'Ferrari', 'points': 356},
        {'driver': 'Oscar Piastri', 'code': 'PIA', 'grid': 4, 'team': 'McLaren', 'points': 292},
        {'driver': 'Carlos Sainz', 'code': 'SAI', 'grid': 5, 'team': 'Ferrari', 'points': 290},
        {'driver': 'George Russell', 'code': 'RUS', 'grid': 6, 'team': 'Mercedes', 'points': 245},
        {'driver': 'Lewis Hamilton', 'code': 'HAM', 'grid': 7, 'team': 'Mercedes', 'points': 223},
        {'driver': 'Sergio Perez', 'code': 'PER', 'grid': 8, 'team': 'Red Bull', 'points': 152},
    ]
    
    # Test 1: Standard Model
    print("\n" + "-" * 80)
    print("1️⃣  STANDARD MODEL (RandomForest)")
    print("-" * 80)
    
    standard_predictor = PreRacePredictor()
    
    print("\n⏳ Training standard model...")
    start_time = time.time()
    standard_success = standard_predictor.train_on_historical_data([2024])
    standard_time = time.time() - start_time
    
    if standard_success:
        print(f"✅ Trained in {standard_time:.1f}s")
        standard_predictions = standard_predictor.predict_race_result(
            qualifying_data, track_type='street'
        )
        
        print("\n📊 Standard Predictions:")
        print(f"{'Pos':<5} {'Driver':<20} {'Grid':<6} {'Predicted':<12} {'Confidence'}")
        print("-" * 70)
        for pred in standard_predictions[:8]:
            print(f"{pred['position']:<5} {pred['driver']:<20} "
                  f"P{pred['grid']:<5} P{pred['predicted_finish']:<11.1f} "
                  f"{pred['confidence']*100:.0f}%")
    else:
        print("❌ Standard model training failed")
        standard_predictions = []
    
    # Test 2: Enhanced Model
    print("\n" + "-" * 80)
    print("2️⃣  ENHANCED MODEL (Ensemble: RF + GB + XGBoost)")
    print("-" * 80)
    
    enhanced_predictor = EnhancedRacePredictor()
    
    print("\n⏳ Training enhanced model...")
    start_time = time.time()
    enhanced_success = enhanced_predictor.train_from_historical_data(
        years=[2024], verbose=False
    )
    enhanced_time = time.time() - start_time
    
    if enhanced_success:
        print(f"✅ Trained in {enhanced_time:.1f}s")
        
        # Get model info
        info = enhanced_predictor.get_model_info()
        print(f"📈 Models: {', '.join(info['models'])}")
        if 'scores' in info:
            print(f"   Position R²: {info['scores']['position']['test_r2']:.3f}")
            print(f"   CV Score: {info['scores']['position']['cv_mean_r2']:.3f} "
                  f"± {info['scores']['position']['cv_std_r2']:.3f}")
        
        enhanced_predictions = enhanced_predictor.predict_race_result(
            qualifying_data, track_name='Monaco'
        )
        
        print("\n📊 Enhanced Predictions (with Confidence Intervals):")
        print(f"{'Pos':<5} {'Driver':<20} {'Grid':<6} {'Predicted':<12} "
              f"{'Range':<15} {'Conf'}")
        print("-" * 80)
        for i, pred in enumerate(enhanced_predictions[:8], 1):
            conf_range = f"P{pred['confidence_lower']:.0f}-P{pred['confidence_upper']:.0f}"
            print(f"{i:<5} {pred['driver']:<20} P{pred['grid']:<5} "
                  f"P{pred['predicted_position']:<11.1f} {conf_range:<15} "
                  f"{pred['confidence']*100:.0f}%")
    else:
        print("❌ Enhanced model training failed")
        enhanced_predictions = []
    
    # Comparison
    print("\n" + "=" * 80)
    print("📊 COMPARISON SUMMARY")
    print("=" * 80)
    
    features = [
        ("Training Time", f"{standard_time:.1f}s", f"{enhanced_time:.1f}s"),
        ("Models Used", "RandomForest", "RF + GB + XGBoost"),
        ("Confidence Intervals", "❌", "✅"),
        ("Driver Form Tracking", "❌", "✅"),
        ("Track-Specific Features", "Basic", "Advanced"),
        ("Cross-Validation", "❌", "✅"),
        ("Ensemble Voting", "❌", "✅"),
        ("Feature Count", "~3", "~24"),
    ]
    
    print(f"\n{'Feature':<30} {'Standard':<20} {'Enhanced':<20}")
    print("-" * 70)
    for feature, standard, enhanced in features:
        print(f"{feature:<30} {standard:<20} {enhanced:<20}")
    
    # Prediction differences
    if standard_predictions and enhanced_predictions:
        print("\n" + "-" * 80)
        print("🔍 PREDICTION DIFFERENCES")
        print("-" * 80)
        
        print(f"\n{'Driver':<20} {'Grid':<6} {'Standard':<12} {'Enhanced':<12} {'Diff'}")
        print("-" * 70)
        
        for i in range(min(len(standard_predictions), len(enhanced_predictions))):
            std_pred = standard_predictions[i]
            enh_pred = next(
                (p for p in enhanced_predictions if p['driver'] == std_pred['driver']),
                None
            )
            
            if enh_pred:
                diff = enh_pred['predicted_position'] - std_pred['predicted_finish']
                diff_str = f"{diff:+.1f}"
                print(f"{std_pred['driver']:<20} P{std_pred['grid']:<5} "
                      f"P{std_pred['predicted_finish']:<11.1f} "
                      f"P{enh_pred['predicted_position']:<11.1f} {diff_str}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)
    print("""
✅ USE ENHANCED MODEL WHEN:
   • You need reliable confidence intervals
   • Track-specific predictions are important
   • You want to track driver form over time
   • Prediction accuracy is critical
   • You have time for initial training (caches afterward)

✅ USE STANDARD MODEL WHEN:
   • You need very fast predictions
   • Simple baseline is sufficient
   • Limited computational resources
   • Quick prototyping

🎯 RECOMMENDED: Enhanced Model for production use
   - More accurate predictions
   - Better uncertainty quantification
   - Richer feature set
   - Cached models load quickly
""")


def main():
    """Run comparison."""
    try:
        compare_predictions()
    except KeyboardInterrupt:
        print("\n\n⚠️  Comparison interrupted")
    except Exception as e:
        print(f"\n❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
