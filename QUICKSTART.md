# Quick Start Guide 🏁

Get started with F1 Race Replay in under 5 minutes!

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
- Press `2` for Historical Replay
- Enter year: `2024`
- Enter GP: `Monaco`
- Wait for data loading (first time takes 1-2 min)
- Enjoy the replay! 🎬

### Option 2: Direct Replay

```bash
# Watch the 2024 Monaco Grand Prix
python main.py --replay --year 2024 --gp Monaco
```

## Controls During Replay

| Key | What It Does |
|-----|--------------|
| **SPACE** | ⏸️ Pause / ▶️ Resume |
| **↑** | 🚀 Speed up (2x, 4x, 8x) |
| **↓** | 🐌 Slow down (0.5x, 0.25x) |
| **→** | ⏩ Jump forward 10 frames |
| **←** | ⏪ Rewind 10 frames |
| **1-4** | Set speed (0.5x, 1x, 2x, 4x) |
| **M** | 🤖 Toggle ML Panel |
| **C** | 💬 Open AI Chat |
| **Click** | 👆 Select driver |

## What You'll See

### On Screen
```
┌─────────────────────────────────────────────────────┐
│  LAP 45/78                    🏁 LIVE STANDINGS     │
│  ⏱ 01:23:45 (x2.0)          ├─ P1. VER  🔴         │
│                               ├─ P2. HAM  +2.3s 🟡  │
│  🌡️ Track: 42°C              ├─ P3. LEC  +4.1s 🔴  │
│  💧 Humidity: 65%             └─ ...                │
│                                                      │
│         ╔════════════╗                              │
│         ║   TRACK    ║   ← Race happening here!    │
│         ║ 🏎️ 🏎️ 🏎️  ║                              │
│         ╚════════════╝                              │
│                                                      │
│  ⌨️ CONTROLS          🤖 ML RACE INSIGHTS           │
│  [SPACE] Pause        ● VER leading by 2.3s        │
│  [←/→] Rewind/Fwd     ● Close battle for P5-P7     │
│  [↑/↓] Speed +/-      ● HAM gaining 0.1s/lap       │
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
# Classic races
python main.py --replay --year 2024 --gp Monaco      # Monaco GP
python main.py --replay --year 2024 --gp Silverstone # British GP
python main.py --replay --year 2024 --gp Spa         # Belgian GP

# Fast playback (2x speed)
python main.py --replay --year 2024 --gp Monaco --speed 2.0

# Predict a 2025 race
python main.py --predict --year 2025 --gp Monaco

# View 2025 schedule
python main.py --schedule
```

## Tips for Best Experience

### First Time Users
1. ✅ Start with Monaco (short track, loads faster)
2. ✅ Use 2x speed for quicker viewing
3. ✅ Click on drivers to see their telemetry
4. ✅ Press `M` to see ML predictions

### Performance Tips
1. 🚀 Don't use `--legacy` flag (optimized is faster)
2. 🚀 Close other apps for better FPS
3. 🚀 Maximize window for better view
4. 🚀 Update GPU drivers if stuttering

### Cool Features to Try
1. 💬 Press `C` and ask "What's DRS?"
2. 📊 Press `T` to see data tables
3. 👆 Click drivers to track them
4. ⚡ Use number keys (1-4) for instant speed change

## Troubleshooting

### "No module named 'arcade'"
```bash
pip install arcade>=3.0.0
```

### "Session not found"
Check GP name spelling or try round number instead:
```bash
python main.py --replay --year 2024 --round 6
```

### Slow loading first time
Normal! FastF1 downloads telemetry data (1-2 min). Subsequent loads are fast (cached).

### Low FPS
1. Close background apps
2. Try smaller window size
3. Reduce speed to 1x

## What's Next?

- 🏆 Try different races and seasons
- 🤖 Explore ML predictions (press `M`)
- 💬 Chat with AI about F1 (press `C`)
- 📊 View detailed tables (press `T`)
- 🎯 Select drivers and track their battle

## Need Help?

- 📖 Full docs: [README.md](README.md)
- 🚀 Performance guide: [PERFORMANCE.md](PERFORMANCE.md)
- 💾 Check [CHANGELOG.md](CHANGELOG.md) for updates
- 🐛 Report issues on GitHub

## API Key (Optional)

For full AI chat features, set your Groq API key:

```bash
# Linux/Mac
export GROQ_API_KEY=your_key_here

# Windows (PowerShell)
$env:GROQ_API_KEY="your_key_here"

# Or create .env file
echo "GROQ_API_KEY=your_key_here" > .env
```

Basic offline Q&A works without a key!

---

**Enjoy the races! 🏎️💨**

If you like this project, ⭐ star it on GitHub!
