# Examples - Enhanced ML Prediction

This directory contains demo scripts showcasing the enhanced ML prediction features.

## Available Examples

### 1. `demo_enhanced_ml.py` - Feature Demonstration

**What it shows**:
- Driver form tracking over multiple races
- Enhanced predictions with confidence intervals
- Track type classification
- Model training and caching

**Run**:
```bash
python examples/demo_enhanced_ml.py
```

**Expected Output**:
```
================================================================================
Demo 1: Driver Form Tracking
================================================================================

📊 Max Verstappen Recent Form:
   Average Position: P1.6
   Average Gain: -0.40 positions
   Consistency: 0.68
   Momentum: -0.60
   Points Rate: 21.2 pts/race

📊 Lando Norris Recent Form (Improving):
   Average Position: P2.4
   Average Gain: 2.20 positions
   Consistency: 0.48
   Momentum: 2.80 (⬆️ Improving!)
   Points Rate: 18.6 pts/race

================================================================================
Demo 2: Enhanced Race Prediction with Confidence Intervals
================================================================================

⏳ Training model on historical data (2022-2024)...
   This may take a few minutes on first run...
   📅 Processing 2023 season...
      Race 5/22...
   ✓ 2023: 380 data points
   📊 Training on 380 samples...
   ✓ Position model: R² = 0.712 (CV: 0.698 ± 0.042)
   ✓ Laptime model: R² = 0.745 (CV: 0.731 ± 0.038)

✅ Model trained successfully!
   Models used: RandomForest, GradientBoosting, XGBoost
   Position R²: 0.712
   Laptime R²: 0.745

------------------------------------------------------------
🔮 Predicting Monaco 2025
------------------------------------------------------------

📊 Predicted Race Results:
Pos   Driver               Grid   Predicted    Confidence
------------------------------------------------------------
1     Max Verstappen       P1     P1.2         P1-P2        92%
2     Charles Leclerc      P2     P2.1         P1-P3        88%
3     Lando Norris         P3     P2.8         P2-P4        85%
...

💡 Key Insights:
   🔥 Oscar Piastri expected to gain ~1 positions!
```

**Duration**: 3-5 minutes first run, <10 seconds cached

---

### 2. `compare_models.py` - Model Comparison

**What it shows**:
- Side-by-side comparison of standard vs enhanced models
- Performance metrics and benchmarks
- Feature comparison table
- Prediction differences

**Run**:
```bash
python examples/compare_models.py
```

**Expected Output**:
```
================================================================================
ML Model Comparison: Standard vs Enhanced
================================================================================

------------------------------------------------------------
1️⃣  STANDARD MODEL (RandomForest)
------------------------------------------------------------

⏳ Training standard model...
✅ Trained in 32.4s

📊 Standard Predictions:
Pos   Driver               Grid   Predicted    Confidence
--------------------------------------------------------------
1     Max Verstappen       P1     P1.5         75%
2     Charles Leclerc      P2     P2.3         72%
...

------------------------------------------------------------
2️⃣  ENHANCED MODEL (Ensemble: RF + GB + XGBoost)
------------------------------------------------------------

⏳ Training enhanced model...
✅ Trained in 187.3s
📈 Models: RandomForest, GradientBoosting, XGBoost
   Position R²: 0.712
   CV Score: 0.698 ± 0.042

📊 Enhanced Predictions (with Confidence Intervals):
Pos   Driver               Grid   Predicted    Range           Conf
--------------------------------------------------------------------------------
1     Max Verstappen       P1     P1.2         P1-P2           92%
2     Charles Leclerc      P2     P2.1         P1-P3           88%
...

================================================================================
📊 COMPARISON SUMMARY
================================================================================

Feature                        Standard            Enhanced            
----------------------------------------------------------------------
Training Time                  32.4s               187.3s              
Models Used                    RandomForest        RF + GB + XGBoost   
Confidence Intervals           ❌                  ✅                  
Driver Form Tracking           ❌                  ✅                  
Track-Specific Features        Basic               Advanced            
Cross-Validation               ❌                  ✅                  
Ensemble Voting                ❌                  ✅                  
Feature Count                  ~3                  ~24                 

🔍 PREDICTION DIFFERENCES
------------------------------------------------------------

Driver               Grid   Standard     Enhanced     Diff
----------------------------------------------------------------------
Max Verstappen       P1     P1.5         P1.2         -0.3
Charles Leclerc      P2     P2.3         P2.1         -0.2
...

💡 RECOMMENDATIONS

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
```

**Duration**: 5-8 minutes first run, <15 seconds cached

---

## Quick Reference

### Import Enhanced Predictor
```python
from src.ml_enhanced import EnhancedRacePredictor
```

### Basic Prediction
```python
predictor = EnhancedRacePredictor()
predictor.train_from_historical_data([2023, 2024])

predictions = predictor.predict_race_result(
    qualifying_data, 
    track_name='Monaco'
)
```

### With Form Tracking
```python
from src.ml_enhanced import DriverFormTracker

tracker = DriverFormTracker(lookback_races=5)
tracker.update('VER', 'Red Bull Racing', 1, 1, 25)

predictor = EnhancedRacePredictor()
predictor.form_tracker = tracker
predictions = predictor.predict_race_result(qualifying_data, 'Monaco')
```

### Get Confidence Intervals
```python
result = predictor.predict_with_confidence(features, 'position')
print(f"Position: P{result['prediction']:.1f}")
print(f"Range: P{result['confidence_lower']:.0f}-P{result['confidence_upper']:.0f}")
print(f"Confidence: {result['confidence_score']:.0%}")
```

## Requirements

### Python Packages
```bash
pip install -r requirements.txt
pip install xgboost  # Optional but recommended
```

### System Requirements
- Python 3.8+
- 4GB+ RAM (for training)
- 1GB+ disk space (for cached models)

## Troubleshooting

### Issue: Import Error
```
ModuleNotFoundError: No module named 'src'
```
**Solution**: Run from project root:
```bash
cd /path/to/F1_GenAI_Strategist
python examples/demo_enhanced_ml.py
```

### Issue: XGBoost Warning
```
⚠️  XGBoost not installed. Install with: pip install xgboost
```
**Solution**: Install XGBoost (optional):
```bash
pip install xgboost
```
The demos will still work without it, using RF + GB only.

### Issue: Training Takes Long
**First run**: 3-5 minutes (trains and caches models)  
**Subsequent runs**: <1 second (loads from cache)

**To speed up first run**:
- Train on fewer years: `train_from_historical_data([2024])`
- Use smaller dataset

### Issue: No Historical Data
```
❌ No training data available
```
**Solution**: Ensure FastF1 cache is populated:
```bash
# Clear cache if corrupted
rm -rf ~/.fastf1
python examples/demo_enhanced_ml.py  # Will re-download
```

## Cache Management

### Cache Location
```
cache/ml_models/
├── enhanced_model_2023_2024.pkl  # Cached model
└── prerace_model_2023_2024.pkl   # Standard model cache
```

### Clear Cache
```bash
# Force retrain by removing cache
rm -rf cache/ml_models/*.pkl
python examples/demo_enhanced_ml.py
```

### Cache Size
- Enhanced model: ~5-10 MB per year
- Typical total: ~20-50 MB for 2-3 years

## Performance Tips

1. **Train once, cache forever**: Models cache automatically
2. **Use multiple years**: `[2022, 2023, 2024]` for better accuracy
3. **Specify track names**: Improves predictions by 5-10%
4. **Update team ratings**: Modify `TEAM_RATINGS` dict for current season
5. **Track driver form**: Use `DriverFormTracker` for season simulations

## Additional Resources

| Resource | Description |
|----------|-------------|
| [ENHANCED_ML.md](../docs/ENHANCED_ML.md) | Complete feature documentation |
| [INTEGRATION_GUIDE.md](../docs/INTEGRATION_GUIDE.md) | Integration instructions |
| [ENHANCEMENTS_SUMMARY.md](../ENHANCEMENTS_SUMMARY.md) | Quick reference |
| [ml_enhanced.py](../src/ml_enhanced.py) | Source code |

## Contributing

To add new examples:
1. Create new `.py` file in this directory
2. Follow existing format (docstring, main function)
3. Update this README with description
4. Test thoroughly

## License

Same as main project (MIT License)
