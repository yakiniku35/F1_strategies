# Integration Guide: Enhanced ML Model

This guide shows how to integrate the enhanced ML model into your existing F1 prediction simulator.

## Quick Integration

### Option 1: Replace Existing Predictor

In `main.py`, replace the standard predictor with the enhanced one:

```python
# OLD
from src.ml_predictor import PreRacePredictor
predictor = PreRacePredictor()
predictor.train_on_historical_data([2023, 2024])

# NEW
from src.ml_enhanced import EnhancedRacePredictor
predictor = EnhancedRacePredictor()
predictor.train_from_historical_data([2023, 2024])
```

### Option 2: Side-by-Side Comparison

Keep both models and show predictions from each:

```python
from src.ml_predictor import PreRacePredictor
from src.ml_enhanced import EnhancedRacePredictor

# Train both
standard = PreRacePredictor()
enhanced = EnhancedRacePredictor()

standard.train_on_historical_data([2024])
enhanced.train_from_historical_data([2024])

# Compare predictions
standard_preds = standard.predict_race_result(qualifying_data)
enhanced_preds = enhanced.predict_race_result(qualifying_data, track_name='Monaco')
```

## Integration into Race Simulator

### Step 1: Update `race_simulator.py`

Add enhanced predictor support:

```python
# At top of file
from src.ml_enhanced import EnhancedRacePredictor

class PredictedRaceSimulator:
    def __init__(self, year: int, gp: str, use_enhanced_ml: bool = True):
        # ... existing code ...
        
        # Add ML predictor
        if use_enhanced_ml:
            self.ml_predictor = EnhancedRacePredictor()
            print("⏳ Training enhanced ML model...")
            self.ml_predictor.train_from_historical_data([2023, 2024])
        else:
            # Use standard model
            from src.ml_predictor import PreRacePredictor
            self.ml_predictor = PreRacePredictor()
            self.ml_predictor.train_on_historical_data([2023, 2024])
```

### Step 2: Use Enhanced Predictions

Update `get_qualifying_results()`:

```python
def get_qualifying_results(self) -> list:
    """Get enhanced ML predictions for qualifying."""
    if self.qualifying_results is not None:
        return self.qualifying_results
    
    # Get baseline driver list
    drivers = self.data_provider.get_predicted_grid(self.year, self.gp_name)
    
    # Prepare for ML prediction
    qualifying_data = [
        {
            'driver': d['name'],
            'code': d['code'],
            'grid': d['grid'],
            'team': d['team'],
        }
        for d in drivers
    ]
    
    # Use enhanced predictor
    if hasattr(self.ml_predictor, 'predict_race_result'):
        # Enhanced predictor
        track_name = self.race_info.get('location') if self.race_info else None
        ml_predictions = self.ml_predictor.predict_race_result(
            qualifying_data, 
            track_name=track_name
        )
        
        # Format with confidence intervals
        results = []
        for i, pred in enumerate(ml_predictions, 1):
            results.append({
                'grid': pred['grid'],
                'code': pred['code'],
                'name': pred['driver'],
                'team': pred['team'],
                'predicted_position': pred['predicted_position'],
                'confidence': pred['confidence'],
                'confidence_range': f"P{pred['confidence_lower']:.0f}-P{pred['confidence_upper']:.0f}",
            })
    else:
        # Standard predictor (fallback)
        ml_predictions = self.ml_predictor.predict_race_result(qualifying_data)
        results = [
            {
                'grid': p['grid'],
                'code': p.get('driver', 'UNK'),
                'name': p['driver'],
                'team': p['team'],
                'predicted_position': p['predicted_finish'],
                'confidence': p['confidence'],
                'confidence_range': 'N/A',
            }
            for p in ml_predictions
        ]
    
    self.qualifying_results = results
    return results
```

### Step 3: Display Confidence in UI

Update prediction display to show confidence ranges:

```python
def display_predictions(predictions):
    """Display predictions with confidence intervals."""
    print("\n🔮 Race Predictions (with Confidence Intervals):")
    print(f"{'Pos':<5} {'Driver':<20} {'Grid':<6} {'Predicted':<12} "
          f"{'Range':<15} {'Confidence'}")
    print("-" * 80)
    
    for i, pred in enumerate(predictions, 1):
        if 'confidence_range' in pred and pred['confidence_range'] != 'N/A':
            # Enhanced prediction
            print(f"{i:<5} {pred['name']:<20} P{pred['grid']:<5} "
                  f"P{pred['predicted_position']:<11.1f} "
                  f"{pred['confidence_range']:<15} {pred['confidence']*100:.0f}%")
        else:
            # Standard prediction
            print(f"{i:<5} {pred['name']:<20} P{pred['grid']:<5} "
                  f"P{pred['predicted_position']:<11.1f} {'N/A':<15} "
                  f"{pred['confidence']*100:.0f}%")
```

## Integration into Main Menu

Update `main.py` to support enhanced predictions:

```python
def predict_future_race(year, gp, speed=1.0, train_model=True, use_enhanced=True):
    """Run a future race prediction and simulation."""
    print(f"\n🔮 預測 {year} {gp} Grand Prix...")
    print(f"   使用模型: {'Enhanced ML' if use_enhanced else 'Standard ML'}")
    print("=" * 50)
    
    # Create simulator with enhanced ML
    simulator = PredictedRaceSimulator(year, gp, use_enhanced_ml=use_enhanced)
    
    # Rest of existing code...
```

Add command-line option:

```python
parser.add_argument('--use-enhanced-ml', action='store_true',
                    help='Use enhanced ML model (default: True)')
parser.add_argument('--use-standard-ml', action='store_true',
                    help='Use standard ML model instead of enhanced')
```

## Driver Form Tracking Integration

### Track driver form over season

```python
from src.ml_enhanced import DriverFormTracker

class SeasonSimulator:
    """Simulate entire season with form tracking."""
    
    def __init__(self):
        self.form_tracker = DriverFormTracker(lookback_races=5)
        self.season_results = []
    
    def simulate_race(self, race_info):
        """Simulate a single race."""
        # ... run race simulation ...
        
        # Update form tracker after race
        for result in race_results:
            self.form_tracker.update(
                driver_code=result['code'],
                team=result['team'],
                position=result['position'],
                qualifying_pos=result['grid'],
                points=result['points']
            )
        
        self.season_results.append(race_results)
    
    def predict_next_race(self, qualifying_data, track_name):
        """Predict next race considering driver form."""
        predictor = EnhancedRacePredictor()
        
        # Use tracked form data
        predictor.form_tracker = self.form_tracker
        
        return predictor.predict_race_result(qualifying_data, track_name)
```

## Arcade Visualization Integration

### Show confidence in overlay

```python
def draw_prediction_overlay(self, predictions):
    """Draw predictions with confidence visualization."""
    y = 100
    
    for pred in predictions[:10]:
        # Driver name and position
        text = f"P{pred['predicted_position']:.1f} {pred['name']}"
        
        # Confidence color (green = high, yellow = medium, red = low)
        if pred['confidence'] > 0.8:
            color = arcade.color.GREEN
        elif pred['confidence'] > 0.6:
            color = arcade.color.YELLOW
        else:
            color = arcade.color.RED
        
        arcade.draw_text(text, 10, y, color, 12)
        
        # Confidence bar
        bar_width = pred['confidence'] * 100
        arcade.draw_rectangle_filled(
            110 + bar_width/2, y, bar_width, 8, color
        )
        
        # Range text
        if 'confidence_range' in pred:
            range_text = pred['confidence_range']
            arcade.draw_text(range_text, 220, y, arcade.color.WHITE, 10)
        
        y += 20
```

## Performance Optimization

### Cache management for faster loads

```python
import os

def ensure_model_cache():
    """Pre-train and cache models for fast startup."""
    predictor = EnhancedRacePredictor()
    
    cache_file = "cache/ml_models/enhanced_model_2023_2024.pkl"
    
    if not os.path.exists(cache_file):
        print("⏳ First-time setup: Training ML models (this takes ~5 minutes)...")
        predictor.train_from_historical_data([2023, 2024])
        print("✅ Models cached for future use!")
    else:
        print("✅ Using cached models (instant load)")
        predictor.train_from_historical_data([2023, 2024])  # Loads from cache

# Call at app startup
if __name__ == "__main__":
    ensure_model_cache()
    main()
```

## Testing Integration

### Test script

```python
def test_enhanced_integration():
    """Test enhanced ML integration."""
    print("Testing Enhanced ML Integration...")
    
    # Test 1: Basic prediction
    predictor = EnhancedRacePredictor()
    success = predictor.train_from_historical_data([2024], verbose=False)
    assert success, "Training failed"
    print("✅ Test 1: Training successful")
    
    # Test 2: Predictions with confidence
    qualifying = [
        {'driver': 'VER', 'code': 'VER', 'grid': 1, 'team': 'Red Bull Racing'}
    ]
    preds = predictor.predict_race_result(qualifying, 'Monaco')
    assert len(preds) > 0, "No predictions"
    assert 'confidence' in preds[0], "Missing confidence"
    assert 'confidence_lower' in preds[0], "Missing confidence range"
    print("✅ Test 2: Predictions with confidence")
    
    # Test 3: Form tracking
    tracker = DriverFormTracker()
    tracker.update('VER', 'Red Bull Racing', 1, 1, 25)
    form = tracker.get_driver_form('VER')
    assert 'momentum' in form, "Missing form metrics"
    print("✅ Test 3: Form tracking works")
    
    print("\n✅ All integration tests passed!")

if __name__ == "__main__":
    test_enhanced_integration()
```

## Migration Checklist

- [ ] Install XGBoost: `pip install xgboost`
- [ ] Import enhanced predictor in relevant files
- [ ] Update predictor initialization
- [ ] Modify prediction calls to include track_name
- [ ] Update UI to display confidence intervals
- [ ] Add form tracking if simulating multiple races
- [ ] Test predictions on known races
- [ ] Cache models for production
- [ ] Update documentation

## Troubleshooting

### Issue: Import Error
```python
# Check if module exists
try:
    from src.ml_enhanced import EnhancedRacePredictor
    print("✅ Enhanced ML available")
except ImportError as e:
    print(f"❌ Import failed: {e}")
```

### Issue: Predictions Different from Standard
**Expected**: Enhanced model uses more features and may give different results.
**Solution**: This is normal. Compare accuracy over multiple races.

### Issue: Slow First Run
**Expected**: Training takes 3-5 minutes first time.
**Solution**: Use cached models (automatic after first run).

## Best Practices

1. **Always specify track_name** for better predictions
2. **Use cached models** in production
3. **Update team ratings** at start of each season
4. **Track driver form** over multiple races
5. **Show confidence intervals** to users
6. **Cross-validate** predictions with actual results

## Support

For issues or questions:
- Check [Enhanced ML Documentation](ENHANCED_ML.md)
- Run demos: `python examples/demo_enhanced_ml.py`
- Compare models: `python examples/compare_models.py`
- Open GitHub issue with details
