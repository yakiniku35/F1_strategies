from src.ml_predictor import PreRacePredictor
import pandas as pd

# 1. 初始化並訓練
predictor = PreRacePredictor()
# 這會花一點時間下載數據
predictor.train_for_season(2023) 

# 2. 假設下一場比賽的排位賽結果 (這是你需要手動輸入的部分，因為未來還沒發生)
# 假設 Verstappen 竿位, Leclerc 第二...
next_race_grid = [
    {'driver': 'VER', 'grid': 1, 'team': 'Red Bull', 'points': 400},
    {'driver': 'LEC', 'grid': 2, 'team': 'Ferrari', 'points': 180},
    {'driver': 'HAM', 'grid': 3, 'team': 'Mercedes', 'points': 170},
    {'driver': 'NOR', 'grid': 4, 'team': 'McLaren', 'points': 150},
    # ... 其他車手
]

# 3. 進行預測
results = predictor.predict_next_race(next_race_grid)

print("=== Next Race Prediction ===")
for i, res in enumerate(results):
    print(f"P{i+1}: {res['driver']} (Model Score: {res['predicted_finish']:.2f})")