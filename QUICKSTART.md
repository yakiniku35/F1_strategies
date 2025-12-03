# Quick Start Guide 🏁

Get started with F1 Race Prediction Simulator in under 5 minutes!

> **📼 Looking for race replays?** Use [f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay) instead.
> This project focuses on **predicting future races**.

## Installation (2 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/yakiniku35/F1_strategies.git
cd F1_strategies

# 2. Install dependencies
pip install -r requirements.txt

# That's it! Ready to go.
```

## First Run (3 minutes)

### Option 1: Interactive Mode (Easiest)

```bash
python main.py
```

Then follow the on-screen menu:
- Press `1` for Race Prediction
- Enter year: `2025`
- Enter GP: `Monaco`
- Watch the AI predict and simulate the race! 🎬

### Option 2: Direct Prediction

```bash
# Predict the 2025 Monaco Grand Prix
python main.py --predict --year 2025 --gp Monaco
```

### Option 3: View Schedule

```bash
# Check the 2025 F1 calendar
python main.py --schedule
```

## Controls During Simulation

| Key | What It Does |
|-----|--------------|
| **SPACE** | ⏸️ Pause / ▶️ Resume |
| **↑** | 🚀 Speed up |
| **↓** | 🐌 Slow down |
| **→** | ⏩ Jump forward |
| **←** | ⏪ Rewind |
| **M** | 🤖 Toggle ML Panel |
| **Click** | 👆 Select driver |

## What You'll See

### Prediction Simulation
```
┌─────────────────────────────────────────────────────┐
│  2025 MONACO GP PREDICTION    🏁 PREDICTED RESULTS  │
│  🤖 AI-Powered Simulation     ├─ P1. VER  🔴        │
│                                ├─ P2. LEC  +1.8s 🔴  │
│  🌡️ Expected: Dry             ├─ P3. NOR  +3.2s 🟠  │
│  🏎️ Grid: Based on 2024       └─ ...                │
│                                                      │
│         ╔════════════╗                              │
│         ║   TRACK    ║   ← Simulation running       │
│         ║ 🏎️ 🏎️ 🏎️  ║                              │
│         ╚════════════╝                              │
│                                                      │
│  🤖 ML RACE INSIGHTS                                │
│  ● VER favored by historical data                   │
│  ● Ferrari strong at Monaco                         │
│  ● Close battle predicted for P3-P5                 │
└─────────────────────────────────────────────────────┘
```

## Common GP Names

Use these names with `--gp` flag:

```
Bahrain, Saudi Arabian, Australia, Azerbaijan, Miami,
Monaco, Spain, Canada, Austria, British, Hungarian,
Belgium, Netherlands, Italy, Singapore, Japan,
Qatar, United States, Mexico, Brazil, Las Vegas, Abu Dhabi
```

## Quick Examples

```bash
# Predict future races
python main.py --predict --year 2025 --gp Monaco
python main.py --predict --year 2025 --gp Silverstone
python main.py --predict --year 2025 --gp Spa

# View 2025 schedule
python main.py --schedule

# Interactive mode
python main.py
```

## For Historical Race Replays

Use the dedicated replay tool instead:

```bash
# Clone and install f1-race-replay
git clone https://github.com/IAmTomShaw/f1-race-replay.git
cd f1-race-replay
pip install -r requirements.txt

# Replay historical races
python main.py --year 2024 --gp Monaco
```

## Tips for Best Experience

### First Time Users
1. ✅ Start with Monaco (interesting track for predictions)
2. ✅ Watch the ML model analyze historical data
3. ✅ Press `M` to see detailed ML predictions
4. ✅ Try different Grand Prix to compare predictions

### Performance Tips
1. 🚀 Close other apps for better performance
2. 🚀 First run downloads historical data (1-2 min)
3. 🚀 Subsequent predictions are faster (cached data)

### Cool Features to Try
1. 📊 Compare predictions across different circuits
2. 🤖 Explore ML insights panel (press `M`)
3. 📅 Check full 2025 calendar (`--schedule`)
4. ⚡ Experiment with different years and tracks

## Troubleshooting

### "No module named 'sklearn'"
```bash
pip install scikit-learn
```

### "Session not found"
- For future races: This is expected! The model predicts based on historical data
- For past races: Try using [f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay) instead

### Slow loading first time
Normal! FastF1 downloads historical telemetry data (1-2 min). Subsequent runs are fast (cached).

## What's Next?

- 🏆 Try predicting different 2025 races
- 🤖 Explore ML prediction insights (press `M`)
- 📊 Compare predictions across circuits
- 📅 Check upcoming race schedule

## Need Help?

- 📖 Full docs: [README.md](README.md)
- 📝 Migration guide: [MIGRATION.md](MIGRATION.md)
- 💾 Check [CHANGELOG.md](CHANGELOG.md) for updates
- 🐛 Report issues on GitHub
- 📼 For replays: [f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay)

---

**Enjoy the races! 🏎️💨**

If you like this project, ⭐ star it on GitHub!
