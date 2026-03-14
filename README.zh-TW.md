# F1 賽事預測模擬器 🏎️

[English](README.md) | **繁體中文**

一個以 Python 建構的 F1 賽事結果預測應用程式，結合機器學習與歷史數據分析，幫助你預見下一場大獎賽的勝負。

---

## 專案簡介

本專案旨在利用 **機器學習模型** 與 F1 官方遙測數據（透過 FastF1），對未來的 Formula 1 賽事進行預測與模擬。你可以：

- 預測各大獎賽的排位結果
- 分析賽道進站策略（提前進站、留守策略等）
- 瀏覽 2025 年 F1 賽程表
- 透過 AI 助理詢問 F1 相關知識

> **📼 尋找歷史賽事回放？** 請使用 [f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay)。  
> 本專案專注於 **預測未來賽事**，而非歷史回放。

---

## 主要功能

### 🔮 賽事預測
利用歷史表現數據，以 AI 模型預測未來大獎賽的排名結果，並以互動式排行榜呈現預測位次、差距與表現指標。

### 🤖 機器學習模型
- **標準 ML 模型：** 根據歷史表現預測車手排位、分析趨勢、提供最佳進站建議，並考量天氣因素。
- **增強版 ML 模型（建議使用）：**
  - 🎯 **集成學習：** 結合 Random Forest、Gradient Boosting 與 XGBoost，提升預測準確度
  - 📊 **車手近況追蹤：** 分析近期表現趨勢、動能與穩定性
  - 🎲 **信心區間：** 提供含不確定性量化的預測範圍
  - 🏁 **賽道專屬特徵：** 依照賽道類型（街道賽、高速賽道、技術性賽道）調整預測
  - ⚡ **智慧快取：** 訓練一次，後續快速載入

### 🎯 策略分析
全面的進站策略優化與比較工具：
- **一停 vs 兩停** 策略比較
- **提前進站（Undercut）/ 留守（Overcut）** 機會偵測
- **油耗策略模擬：** 逐圈計算油重對圈速的影響
- **賽道專屬輪胎退化分析：** 高壓賽道（銀石、鈴鹿）採保守策略；低壓賽道（蒙扎、巴林）採積極策略

### 📅 賽程表
檢視 2025 年 F1 完整賽程與賽事資訊。

---

## 系統需求

- Python 3.8 以上
- [FastF1](https://github.com/theOehrly/Fast-F1) — F1 遙測數據
- [Arcade](https://api.arcade.academy/) — 圖形介面函式庫
- scikit-learn — 機器學習
- xgboost — 增強版 ML 模型

---

## 安裝方式

1. 複製儲存庫：
   ```bash
   git clone https://github.com/yakiniku35/F1_strategies.git
   cd F1_strategies
   ```

2. 安裝相依套件：
   ```bash
   pip install -r requirements.txt
   ```

3. （選用）設定 AI 評論功能的環境變數：
   ```bash
   echo "GROQ_API_KEY=your_api_key_here" > .env
   ```

---

## 使用方式

### 互動模式（建議）

```bash
python main.py
```

啟動後會出現選單：
1. 🔮 預測未來賽事
2. 📼 回放歷史賽事
3. 📅 查看賽程表
4. 🎯 策略分析
5. ❌ 離開

### 命令列模式

```bash
# 預測 2025 年摩納哥大獎賽
python main.py --predict --year 2025 --gp Monaco

# 查看 2025 年賽程
python main.py --schedule

# 執行銀石賽道策略分析（共 52 圈）
python main.py --strategy --track Silverstone --laps 52
```

### 指令參數說明

| 參數 | 說明 |
|------|------|
| `--predict` | 預測並模擬未來賽事 |
| `--schedule` | 查看 2025 F1 賽程表 |
| `--strategy` | 執行賽事策略分析 |
| `--year` | 賽事年份（預設：2025） |
| `--gp` | 大獎賽名稱（如 Monaco、Silverstone） |
| `--round` | 輪次編號（可替代 `--gp`） |
| `--track` | 策略分析的賽道名稱 |
| `--laps` | 策略分析的總圈數（預設：50） |
| `--speed` | 初始播放速度（預設：1.0） |
| `--no-train` | 跳過 ML 模型訓練（預測模式） |

---

## 模擬畫面預覽

```
┌─────────────────────────────────────────────────────┐
│  2025 MONACO GP PREDICTION    🏁 PREDICTED RESULTS  │
│  🤖 AI-Powered Simulation     ├─ P1. VER  🔴        │
│                                ├─ P2. LEC  +1.8s 🔴  │
│  🌡️ Expected: Dry             ├─ P3. NOR  +3.2s 🟠  │
│  🏎️ Grid: Based on 2024       └─ ...                │
│                                                      │
│         ╔════════════╗                              │
│         ║   TRACK    ║   ← 模擬進行中               │
│         ║ 🏎️ 🏎️ 🏎️  ║                              │
│         ╚════════════╝                              │
│                                                      │
│  🤖 ML RACE INSIGHTS                                │
│  ● VER 歷史數據最佳                                  │
│  ● 法拉利在摩納哥表現強勁                            │
│  ● P3–P5 預測將有激烈爭奪                            │
└─────────────────────────────────────────────────────┘
```

---

## 專案結構

```
F1_strategies/
├── main.py                 # 主程式入口（互動模式）
├── run_integrated.py       # 整合管線入口
├── src/
│   ├── arcade_replay.py    # Arcade 視覺化與 UI
│   ├── f1_data.py          # 遙測資料載入與處理
│   ├── ml_predictor.py     # 機器學習預測
│   ├── ml_enhanced.py      # 增強版 ML 模型
│   ├── ai_chat.py          # AI 聊天助理
│   └── strategy_analyzer.py # 策略分析引擎
├── docs/                   # 詳細技術文件
├── examples/               # 範例腳本
├── images/
│   └── tyres/              # 輪胎圖示
├── data/
│   └── track_layouts/      # 賽道佈局快取
└── requirements.txt
```

---

## 常見問題排解

### Arcade 安裝問題（Linux）
```bash
sudo apt-get install python3-dev libgl1-mesa-dev
pip install arcade
```

### FastF1 快取問題
```bash
rm -rf .fastf1-cache/
python main.py --year 2023 --gp Monaco --refresh-data
```

### 找不到 sklearn 模組
```bash
pip install scikit-learn
```

---

## 技術致謝

- 靈感來源：[f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay)（Tom Shaw 作品）
- 數據來源：[FastF1](https://github.com/theOehrly/Fast-F1)
- 圖形引擎：[Arcade](https://api.arcade.academy/)

---

## ⚠️ 免責聲明

Formula 1 及相關商標為各自所有人之財產。所有資料均來自公開 API，僅供教育用途。

## 授權條款

MIT License
