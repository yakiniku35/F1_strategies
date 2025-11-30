# F1 Race Replay with ML Prediction 🏎️

A Python application for visualizing Formula 1 race telemetry with real-time machine learning predictions. Built with Arcade graphics library for smooth performance.

## Features

- **Race Replay Visualization:** Watch races unfold with real-time driver positions on a rendered track
- **Machine Learning Predictions:** AI-powered predictions for race trends, position changes, and battles
- **Interactive Leaderboard:** See live driver positions and current tyre compounds
- **Driver Telemetry:** View speed, gear, DRS status, and lap information for selected drivers
- **Track Status Indicators:** Yellow flags, Safety Car, Virtual Safety Car, Red flags
- **Playback Controls:** Pause, rewind, fast forward, and adjust playback speed

## Controls

| Key | Action |
|-----|--------|
| SPACE | Pause/Resume |
| ← / → | Rewind / Fast Forward |
| ↑ / ↓ | Increase / Decrease Speed |
| 1-4 | Set Speed (0.5x, 1x, 2x, 4x) |
| M | Toggle ML Prediction Panel |
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

Run the replay with a specific race:

```bash
# By Grand Prix name
python main.py --year 2023 --gp Monaco

# By round number
python main.py --year 2023 --round 7

# With custom options
python main.py --year 2024 --gp Silverstone --speed 2.0

# Force refresh telemetry data
python main.py --year 2023 --gp Monaco --refresh-data
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--year` | Race year (default: 2023) |
| `--gp` | Grand Prix name (e.g., Monaco, Silverstone) |
| `--round` | Round number (alternative to --gp) |
| `--session` | Session type: R, Q, FP1, FP2, FP3 (default: R) |
| `--speed` | Initial playback speed (default: 1.0) |
| `--refresh-data` | Force refresh of telemetry data |

## Machine Learning Features

The application includes ML-powered race analysis:

- **Position Prediction:** Predicts future driver positions based on current pace
- **Trend Analysis:** Identifies improving/declining drivers
- **Battle Detection:** Predicts upcoming on-track battles
- **Strategy Insights:** Pit window recommendations based on tyre degradation

The ML model trains on race data during initialization and provides real-time insights during replay.

## Project Structure

```
F1_strategies/
├── main.py                 # Entry point
├── src/
│   ├── arcade_replay.py    # Visualization and UI logic
│   ├── f1_data.py          # Telemetry loading and processing
│   ├── ml_predictor.py     # Machine learning predictions
│   └── lib/
│       └── tyres.py        # Tyre compound utilities
├── images/
│   └── tyres/              # Tyre compound icons
├── resources/              # Background images
├── computed_data/          # Cached telemetry data
└── requirements.txt
```

## Legacy Streamlit Version

The original Streamlit-based version files are still available:
- `app.py` - Streamlit application (deprecated)
- `utils.py` - Legacy utilities
- `data_loader.py` - Legacy data loader
- `llm_helper.py` - LLM helper functions

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