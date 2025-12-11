# F1 Strategy Analysis Guide 🎯

## Overview

The F1 Strategy Analyzer provides comprehensive race strategy tools to help simulate and optimize pit stop strategies, detect undercut/overcut opportunities, and analyze fuel management.

## Quick Start

### Command Line Usage

Run strategy analysis from the command line:

```bash
# Analyze strategies for a specific track
python main.py --strategy --track Silverstone --laps 52

# Different tracks with varying lap counts
python main.py --strategy --track Monaco --laps 78
python main.py --strategy --track Monza --laps 53
python main.py --strategy --track Spa --laps 44
```

### Interactive Mode

Launch the interactive strategy analyzer:

```bash
python main.py
# Select option 4: 🎯 策略分析 (Strategy Analysis)
```

## Strategy Features

### 1. Strategy Comparison

Compare multiple pit stop strategies for any track and race distance:

**Available Strategies:**
- **Aggressive One-Stop**: Early pit (lap ~30% of race), push on softs then manage hards
- **Conservative One-Stop**: Later pit (lap ~45%), balanced approach with mediums
- **Two-Stop Sprint**: Maximum pace with fresh tyres, pits at ~25% and ~60%
- **Medium-Hard One-Stop**: Safe strategy starting on mediums, switching to hards
- **Undercut Special**: Early pit for track position (only for top 5 positions)

**Example Output:**
```
╔══════════════════════════════════════════════════════════════╗
║              STRATEGY COMPARISON                             ║
╠══════════════════════════════════════════════════════════════╣
║ 1. Conservative One-Stop     ║
║    Stops: 1  |  Risk: LOW                ║
║    Compounds: MEDIUM → MEDIUM                      ║
║    Pit Laps: 23                                      ║
║    Est. Time: 78:54     (+0.0s)         ║
║    Balanced approach, pit lap 23 for fresh mediums        ║
╠══════════════════════════════════════════════════════════════╣
```

### 2. Undercut Analysis

Analyze if pitting before your rival is advantageous:

**Factors Considered:**
- Gap to car ahead (ideal: < 5 seconds)
- Tyre age advantage
- Current lap vs. optimal pit window

**Example:**
```
Current lap: 20
Gap to car ahead: 3.5s
Our tyre age: 10 laps
Their tyre age: 15 laps

Result:
🟢 STRONG UNDERCUT: Box this lap! Gap 3.5s is ideal
Score: 0.78 / 1.00
✓ Gap advantage
✓ Tyre advantage
✓ In pit window
```

### 3. Overcut Analysis

Determine if staying out longer can gain positions:

**Factors Considered:**
- Current tyre compound and remaining life
- Tyre health status
- Optimal stint length for compound

**Example:**
```
Current lap: 25
Gap to car ahead: 5.0s
Our tyre age: 15 laps
Current compound: MEDIUM

Result:
🟢 STAY OUT: Extend 10 more laps, build gap on fresh tyres
Remaining optimal laps: 10
✓ Tyres healthy
Recommended extend: 5 laps
```

### 4. Fuel Strategy Simulation

Optimize fuel management for race performance:

**Analysis Includes:**
- Fuel consumption per lap
- Lap time penalty from fuel weight
- Optimal push periods when car is lightest
- Lap-by-lap fuel weight impact

**Example:**
```
Starting fuel: 110 kg
Total laps: 50

Results:
- Fuel per lap: 2.20 kg
- Initial penalty: 3.30 s/lap
- Final penalty: 0.20 s/lap
- Best performance window: Lap 35-50
- Car will be 77.0kg lighter at lap 35
```

### 5. Full Strategy Report

Get comprehensive strategy summary for current race situation:

**Includes:**
- Current situation analysis (lap, position, tyre health)
- Top 3 recommended strategies
- Next action recommendation
- Pit urgency level

## Track-Specific Characteristics

The analyzer adapts strategies based on track type:

### High Tyre Stress Tracks
- **Silverstone**: High-speed corners, aggressive tyre wear
- **Suzuka**: Technical sections, high degradation
- **Barcelona**: Hot temperatures, conservative strategies recommended
- **Spa**: Fast corners, extended pit windows

**Strategy Impact**: More conservative, longer stints on harder compounds

### Low Tyre Stress Tracks
- **Monza**: Low downforce, minimal wear
- **Bahrain**: Smooth surface, gentle on tyres
- **Austria**: Short lap, multiple strategy options

**Strategy Impact**: Aggressive strategies viable, softer compounds beneficial

### Street Circuits
- **Monaco**: Narrow, overtaking difficult
- **Singapore**: Night race, unique characteristics
- **Jeddah**: High-speed street circuit

**Strategy Impact**: Track position critical, undercut strategies crucial

## Tyre Compound Characteristics

### Soft Compound
- **Pace**: Fastest (baseline lap time)
- **Degradation**: 0.05s per lap
- **Optimal Stint**: 10-20 laps
- **Use Case**: Qualifying, short stints, undercuts

### Medium Compound
- **Pace**: 0.3s slower than soft
- **Degradation**: 0.02s per lap
- **Optimal Stint**: 20-30 laps
- **Use Case**: Balanced race strategies, one-stops

### Hard Compound
- **Pace**: 0.6s slower than soft
- **Degradation**: 0.01s per lap
- **Optimal Stint**: 30-45 laps
- **Use Case**: Long stints, high-degradation tracks, safety car situations

## Export Strategies

Save strategy analysis for later reference:

### JSON Export
```bash
# In interactive mode, after viewing strategies
匯出策略? Export strategies? (y=JSON/c=CSV/n=No): y
✅ Strategies exported to: strategy_Silverstone_52laps.json
```

### CSV Export
```bash
匯出策略? Export strategies? (y=JSON/c=CSV/n=No): c
✅ Strategies exported to: strategy_Silverstone_52laps.csv
```

## Integration with Race Prediction

When using race prediction mode, strategy recommendations are automatically displayed:

```bash
python main.py --predict --year 2025 --gp Monaco --no-train
```

Output includes:
1. Qualifying predictions
2. **Strategy recommendations for pole position driver**
3. Option to launch race simulation

## Programmatic Usage

Use the strategy analyzer in your own Python scripts:

```python
from src.strategy_analyzer import StrategyAnalyzer

# Initialize analyzer
analyzer = StrategyAnalyzer(track_name="Silverstone", total_laps=52)

# Generate strategy options
strategies = analyzer.generate_strategy_options(current_position=5)

# Compare strategies
comparison = analyzer.compare_strategies(strategies)
print(comparison)

# Undercut analysis
undercut = analyzer.analyze_undercut_opportunity(
    current_lap=20,
    gap_to_car_ahead=3.5,
    our_tyre_age=10,
    their_tyre_age=15
)
print(undercut['recommendation'])

# Overcut analysis
overcut = analyzer.analyze_overcut_opportunity(
    current_lap=25,
    gap_to_car_ahead=5.0,
    our_tyre_age=15,
    our_compound="MEDIUM"
)
print(overcut['recommendation'])

# Fuel strategy
fuel = analyzer.simulate_fuel_strategy(fuel_load=110.0)
print(f"Fuel per lap: {fuel['fuel_per_lap']:.2f} kg")

# Full strategy report
summary = analyzer.get_strategy_summary(
    current_lap=20,
    current_position=5,
    current_tyre="MEDIUM",
    tyre_age=10
)
print(summary['next_action'])

# Export strategies
analyzer.export_strategies_to_json(strategies, "my_strategies.json")
analyzer.export_strategies_to_csv(strategies, "my_strategies.csv")
```

## Tips & Best Practices

### When to Use Each Strategy

1. **Aggressive One-Stop**
   - Leading from front
   - Need to respond to competitor's early stop
   - Track position critical (Monaco, Hungary)

2. **Conservative One-Stop**
   - Midfield battles
   - Uncertain weather conditions
   - Minimizing risk in points-paying positions

3. **Two-Stop Sprint**
   - Starting from back of grid
   - Need to overtake many cars
   - Low-degradation tracks (Bahrain, Monza)
   - Fresh tyre advantage outweighs pit time loss

4. **Undercut Special**
   - Stuck behind slower car
   - Tyre advantage over competitor
   - DRS train situations

### Undercut Strategy

**Best Situations:**
- Gap to car ahead: 2-5 seconds
- Your tyres 5+ laps fresher than opponent
- Within optimal pit window (laps 30-60% of race)

**Execution:**
1. Build up to car ahead in final laps before pit
2. Pit 2-3 laps before opponent's optimal window
3. Push hard on fresh tyres (out-laps)
4. Gain track position when they pit

### Overcut Strategy

**Best Situations:**
- Opponent has pitted
- Your tyres still have 5+ laps of life
- Can maintain pace on current tyres
- Building gap while they're in traffic

**Execution:**
1. Stay out when opponent pits
2. Push to maximize gap (clear air advantage)
3. Pit when tyres reach cliff
4. Rejoin ahead due to time gained

### Fuel Management

**Early Race (Laps 1-30%):**
- Heavy fuel load (~3s penalty per lap)
- Focus on tyre management over pace
- Avoid unnecessary battles

**Mid Race (Laps 30-70%):**
- Fuel load decreasing
- Optimal balance of pace and conservation
- Execute pit stops during this window

**Late Race (Laps 70-100%):**
- Lightest fuel load (<1s penalty)
- Maximum attack possible
- Ideal for overtaking attempts

## Troubleshooting

### Issue: Strategy times seem incorrect
**Solution**: Ensure correct track and lap count. Different tracks have different base lap times and characteristics.

### Issue: Undercut showing not viable when it should be
**Solution**: Check all three factors: gap (<5s), tyre advantage (5+ laps), and pit window (30-60% of race).

### Issue: Export not working
**Solution**: Ensure you have write permissions in the current directory. Files are saved in the working directory.

## Advanced Topics

### Custom Strategy Development

Modify `strategy_analyzer.py` to add custom strategies:

```python
# Example: Add a "Safety Car Special" strategy
if has_safety_car:
    strategies.append(StrategyOption(
        name="Safety Car Special",
        stops=1,
        pit_laps=[safety_car_lap],
        compounds=["SOFT", "HARD"],
        estimated_time=self._estimate_strategy_time(
            [safety_car_lap], ["SOFT", "HARD"]
        ),
        risk_level="MEDIUM",
        description=f"Pit under safety car lap {safety_car_lap}, save time"
    ))
```

### Track-Specific Tuning

Adjust tyre degradation rates for specific tracks:

```python
# In strategy_analyzer.py
TRACK_TYRE_STRESS = {
    "Monaco": 0.8,    # Lower degradation (slow corners)
    "Silverstone": 1.3,  # Higher degradation (fast corners)
    # Add more tracks as needed
}
```

## Resources

- **FastF1 Documentation**: https://docs.fastf1.dev/
- **F1 Technical Regulations**: https://www.fia.com/regulation/category/110
- **Tyre Strategy Analysis**: Historical F1 race data
- **Code Repository**: Check `src/strategy_analyzer.py` for implementation details

## Support

For issues, questions, or feature requests:
1. Check this guide first
2. Review code comments in `src/strategy_analyzer.py`
3. Open an issue on GitHub
4. Test with command line mode first (`--strategy`)

---

**Note**: Strategy recommendations are estimates based on typical F1 race conditions. Actual race scenarios may vary due to weather, safety cars, incidents, and driver performance.
