# F1 Race Replay with ML Prediction 🏎️

A Python application for visualizing Formula 1 race telemetry with real-time machine learning predictions. Built with Arcade graphics library for ultra-smooth 60 FPS performance with advanced frame interpolation.

## Features

- **Smooth Race Replay:** Watch races unfold with buttery-smooth 60 FPS rendering and sub-frame interpolation
- **Optimized Performance:** NumPy-based batch rendering for fluid animation even with 20+ cars
- **Machine Learning Predictions:** AI-powered predictions for race trends, position changes, and battles
- **Interactive Leaderboard:** See live driver positions, gaps, intervals, and current tyre compounds
- **Driver Telemetry:** View speed, gear, DRS status, and lap information for selected drivers
- **Track Status Indicators:** Yellow flags, Safety Car, Virtual Safety Car, Red flags
- **Playback Controls:** Pause, rewind, fast forward, and adjust playback speed (0.5x to 8x)
- **AI Chat Assistant:** Ask questions about F1 strategy, rules, and current race situation 🤖

## Performance Features

- **60 FPS Rendering:** VSync-enabled smooth animation
- **Frame Interpolation:** Sub-frame position interpolation for ultra-smooth car movement
- **Batch Rendering:** OpenGL sprite batching reduces CPU overhead
- **NumPy Optimization:** Fast array operations for large datasets
- **Cached Track Shapes:** Track geometry pre-calculated and reused

## Controls

| Key | Action |
|-----|--------|
| SPACE | Pause/Resume |
| ← / → | Rewind / Fast Forward |
| ↑ / ↓ | Increase / Decrease Speed |
| 1-4 | Set Speed (0.5x, 1x, 2x, 4x) |
| M | Toggle ML Prediction Panel |
| T | Toggle Tables View |
| C | Open AI Chat (ask questions!) 🤖 |
| ESC | Close chat panel |
| Click | Select driver on leaderboard |

## Requirements

- Python 3.8+
- [FastF1](https://github.com/theOehrly/Fast-F1) - F1 telemetry data
- [Arcade](https://api.arcade.academy/) - Graphics library
- scikit-learn - Machine learning

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yakiniku35/F1_strategies.git
   cd F1_strategies
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up environment variables for AI commentary:
   ```bash
   # Create .env file
   echo "GROQ_API_KEY=your_api_key_here" > .env
   ```

## Usage

### Interactive Mode (Recommended)

Run without arguments for an interactive menu:

```bash
python main.py
```

### Command Line Mode

Run the replay with a specific race:

```bash
# By Grand Prix name (optimized format - faster)
python main.py --replay --year 2024 --gp Monaco

# By round number
python main.py --replay --year 2024 --round 7

# Predict a future race
python main.py --predict --year 2025 --gp Monaco

# View 2025 schedule
python main.py --schedule

# Custom playback speed
python main.py --replay --year 2024 --gp Silverstone --speed 2.0

# Use legacy format (slower, for compatibility)
python main.py --replay --year 2023 --gp Monaco --legacy
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--predict` | Predict and simulate a future race |
| `--replay` | Replay a historical race |
| `--schedule` | View 2025 F1 calendar |
| `--year` | Race year (default: 2024) |
| `--gp` | Grand Prix name (e.g., Monaco, Silverstone) |
| `--round` | Round number (alternative to --gp) |
| `--speed` | Initial playback speed (default: 1.0, max: 8.0) |
| `--legacy` | Use legacy dict format instead of optimized NumPy arrays |
| `--no-train` | Skip ML model training (prediction mode) |

## Performance Tips

1. **Use Optimized Format:** By default, the app uses NumPy arrays for 5-10x better performance
2. **Adjust Speed:** Use `--speed` to control initial playback (1x = real-time, 2x = double speed)
3. **Hardware Acceleration:** Ensure GPU drivers are up-to-date for best OpenGL performance
4. **Cache Directory:** First load is slower (downloading data), subsequent loads are fast (cached)

## Machine Learning Features

The application includes ML-powered race analysis:

- **Position Prediction:** Predicts future driver positions based on current pace
- **Trend Analysis:** Identifies improving/declining drivers
- **Battle Detection:** Predicts upcoming on-track battles
- **Strategy Insights:** Pit window recommendations based on tyre degradation

The ML model trains on race data during initialization and provides real-time insights during replay.

## AI Chat Feature 🤖

Press **C** to open the AI Chat panel and ask questions about F1:

- "What's the best tyre strategy?"
- "Who's likely to win?"
- "Explain DRS"
- "What's a good pit window?"

The AI assistant provides context-aware responses based on the current race state.

**Note:** For full AI chat functionality, set your Groq API key:
```bash
export GROQ_API_KEY=your_api_key_here
```

Basic offline Q&A is available without an API key.

## Project Structure

```
F1_strategies/
├── main.py                 # Main entry point (interactive mode)
├── run_integrated.py       # Integration pipeline entry point
├── src/
│   ├── arcade_replay.py    # Arcade visualization and UI
│   ├── f1_data.py          # Telemetry loading and processing
│   ├── ml_predictor.py     # Machine learning predictions
│   ├── ai_chat.py          # AI chat assistant for Q&A
│   ├── dashboard/
│   │   ├── prediction_overlay.py  # ML prediction overlay
│   │   └── tables.py       # Data tables generation
│   ├── integration/
│   │   └── pipeline.py     # Full analysis pipeline
│   ├── simulation/
│   │   ├── race_simulator.py      # Race simulation
│   │   ├── future_race_data.py    # Future race data provider
│   │   └── track_layouts.py       # Track layout manager
│   └── lib/
│       └── tyres.py        # Tyre compound utilities
├── images/
│   └── tyres/              # Tyre compound icons
├── data/
│   └── track_layouts/      # Cached track layouts
└── requirements.txt
```

## Troubleshooting

### Arcade installation issues

If you encounter issues installing Arcade on Linux:
```bash
sudo apt-get install python3-dev libgl1-mesa-dev
pip install arcade
```

### FastF1 cache issues

If telemetry loading is slow or fails:
```bash
# Clear the cache and retry
rm -rf .fastf1-cache/
python main.py --year 2023 --gp Monaco --refresh-data
```

## Credits

- Inspired by [f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay) by Tom Shaw
- Data from [FastF1](https://github.com/theOehrly/Fast-F1)
- Graphics powered by [Arcade](https://api.arcade.academy/)

## ⚠️ Disclaimer

Formula 1 and related trademarks are the property of their respective owners. All data is sourced from publicly available APIs and used for educational purposes only.

## License

MIT License