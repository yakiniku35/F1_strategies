# Migration Guide - Historical Race Replay Removed

## What Changed?

As of this update, **historical race replay functionality has been removed** from this project. This project now focuses exclusively on **race prediction and simulation** for future races.

## Why?

Instead of maintaining duplicate replay functionality, we recommend using the excellent dedicated replay tool:

**[f1-race-replay by IAmTomShaw](https://github.com/IAmTomShaw/f1-race-replay)**

This project provides:
- Smooth 60 FPS race visualization
- Interactive controls and playback speed adjustment
- Detailed telemetry displays
- Track status indicators
- Dedicated focus on replay features

## What This Project Now Does

This project focuses on:
- 🔮 **Predicting future races** using ML models
- 📊 **Analyzing race strategies** and outcomes
- 📅 **Viewing race calendars** and schedules

## Migration Instructions

### Old Commands (No Longer Work)

```bash
# ❌ These commands have been removed:
python main.py --replay --year 2024 --gp Monaco
python main.py --replay --year 2024 --round 7
python main.py --replay --year 2024 --gp Monaco --legacy
```

### New Commands (Still Work)

```bash
# ✅ Predict a future race
python main.py --predict --year 2025 --gp Monaco

# ✅ View race schedule
python main.py --schedule

# ✅ Interactive mode (option 2 removed)
python main.py
```

### For Historical Race Replays

Install and use [f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay):

```bash
# Install the dedicated replay tool
git clone https://github.com/IAmTomShaw/f1-race-replay.git
cd f1-race-replay
pip install -r requirements.txt

# Replay a race
python main.py --year 2024 --gp Monaco
```

## Interactive Menu Changes

**Before:**
```
1. 🔮 預測未來比賽 (Predict Future Race)
2. 📼 回放歷史比賽 (Replay Historical Race)  ← REMOVED
3. 📅 查看賽程表 (View Schedule)
4. ❌ 離開 (Exit)
```

**After:**
```
1. 🔮 預測未來比賽 (Predict Future Race)
2. 📅 查看賽程表 (View Schedule)
3. ❌ 離開 (Exit)
```

## Questions?

- For race **prediction** features: Continue using this repository
- For race **replay** features: Use [f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay)

---

*This change keeps our codebase focused and leverages existing specialized tools in the F1 community.*
