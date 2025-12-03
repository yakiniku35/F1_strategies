# F1 Race Prediction Simulator 🏎️

A Python application for predicting Formula 1 race outcomes using machine learning and historical data analysis.

## Features

- **Race Prediction:** AI-powered predictions for future race outcomes using historical performance data
- **Machine Learning Models:** Advanced ML models analyzing driver performance, track conditions, and historical trends
- **Interactive Leaderboard:** View predicted positions, gaps, and performance metrics
- **Schedule Viewer:** Check upcoming race calendars and event information
- **Data-Driven Insights:** Predictions based on FastF1 telemetry and historical race data

## Historical Race Replay

**For watching historical race replays**, we recommend using the excellent [f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay) project by IAmTomShaw. It provides a dedicated, feature-rich replay system with:
- Smooth 60 FPS race visualization
- Interactive controls and playback speed adjustment
- Detailed telemetry displays
- Track status indicators

This project focuses on **prediction and simulation** of future races rather than historical replay functionality.

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

You'll see a menu with options to:
1. 🔮 Predict future races
2. 📅 View race schedule
3. ❌ Exit

### Command Line Mode

Predict a future race or view the schedule:

```bash
# Predict a future race
python main.py --predict --year 2025 --gp Monaco

# View 2025 schedule
python main.py --schedule

# Predict with custom speed
python main.py --predict --year 2025 --gp Silverstone --speed 2.0
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--predict` | Predict and simulate a future race |
| `--schedule` | View 2025 F1 calendar |
| `--year` | Race year (default: 2025) |
| `--gp` | Grand Prix name (e.g., Monaco, Silverstone) |
| `--round` | Round number (alternative to --gp) |
| `--speed` | Initial playback speed (default: 1.0) |
| `--no-train` | Skip ML model training (prediction mode) |

## Machine Learning Features

The application includes ML-powered race prediction:

- **Position Prediction:** Predicts future driver positions based on historical performance
- **Trend Analysis:** Identifies driver form and performance trends
- **Strategy Insights:** Analyzes optimal pit stop windows and tyre strategies
- **Weather Impact:** Considers weather conditions in race predictions

The ML model trains on historical race data and provides predictions for upcoming races.
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