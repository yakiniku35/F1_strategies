# ML 模型快取功能

## 功能說明

為了避免每次預測都需要重新訓練模型，系統現在支援模型快取功能。訓練完成的模型會自動儲存，下次使用時會直接從快取載入。

## 快取位置

模型快取儲存在 `cache/ml_models/` 目錄下：

```
cache/
└── ml_models/
    ├── prerace_model_2023_2024.pkl   # 預測模型 (2023-2024 資料)
    └── ...
```

## 快取機制

### 1. 預測模式 (PreRacePredictor)

- **快取檔名**: `prerace_model_{years}.pkl`
- **快取鍵值**: 根據訓練年份生成 (例如: `prerace_model_2023_2024`)
- **自動快取**: 訓練完成後自動儲存
- **自動載入**: 下次使用時自動檢查並載入快取

### 2. 歷史模式 (RaceTrendPredictor)

- **快取檔名**: `{cache_key}.pkl`
- **快取鍵值**: 根據資料特徵生成 MD5 hash
- **手動快取**: 需要手動呼叫 `save_model()` 和 `load_model()`

## 使用範例

### 預測模式 (自動快取)

```python
from src.ml_predictor import PreRacePredictor

predictor = PreRacePredictor()

# 第一次: 訓練並自動儲存
predictor.train_on_historical_data([2023, 2024])
# 輸出: ✅ 模型已儲存至快取

# 第二次: 自動從快取載入
predictor.train_on_historical_data([2023, 2024])
# 輸出: ✅ 從快取載入模型 (訓練年份: 2023, 2024)
```

### 歷史模式 (手動快取)

```python
from src.ml_predictor import RaceTrendPredictor

predictor = RaceTrendPredictor()

# 訓練模型
predictor.train(frames, drivers)

# 手動儲存
cache_key = "my_race_model"
predictor.save_model(cache_key)

# 下次載入
predictor.load_model(cache_key)
```

## 快取管理

### 清除所有快取

```bash
rm -rf cache/ml_models/*.pkl
```

### 查看快取檔案

```bash
ls -lh cache/ml_models/
```

### 快取檔案大小

每個模型約 10-50 MB，包含:
- 訓練好的隨機森林模型
- 資料標準化器 (Scaler)
- 模型元資料

## 效能提升

使用快取後:
- ⚡ **首次運行**: ~2-5 分鐘 (需訓練)
- ⚡ **後續運行**: ~2-5 秒 (快取載入)
- 📉 **時間節省**: 約 95-99%

## 注意事項

1. **快取失效**: 當訓練資料年份改變時，會自動重新訓練
2. **磁碟空間**: 確保有足夠空間儲存模型 (每個約 50 MB)
3. **版本相容**: 模型使用 pickle 序列化，確保 Python 版本相容
4. **安全性**: 不要載入來源不明的 .pkl 檔案

## Troubleshooting

### 問題: 無法載入快取

```
⚠️  無法載入模型: ...
```

**解決方法**:
1. 刪除損壞的快取檔案
2. 重新訓練模型

### 問題: 快取佔用空間過大

**解決方法**:
```bash
# 只保留最新的快取
cd cache/ml_models/
ls -t *.pkl | tail -n +2 | xargs rm
```

## 技術細節

### 快取內容

**PreRacePredictor**:
```python
{
    'model': RandomForestRegressor,
    'scaler': StandardScaler,
    'is_trained': bool,
    'training_years': list
}
```

**RaceTrendPredictor**:
```python
{
    'position_model': RandomForestRegressor,
    'laptime_model': GradientBoostingRegressor,
    'scaler': StandardScaler,
    'is_trained': bool
}
```

### 快取策略

- **寫入**: 訓練完成後立即寫入
- **讀取**: 訓練前先嘗試讀取
- **過期**: 根據快取鍵值自動判斷

---

更新時間: 2024-12-12
