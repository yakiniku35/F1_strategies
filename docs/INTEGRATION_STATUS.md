# F1-Race-Replay 整合狀態

## ✅ 整合完成

您的 F1_strategies 專案已成功整合 [f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay) 項目！

## 📂 整合架構

```
F1_strategies/
├── src/
│   ├── external_f1_data.py          # 外部 replay 數據處理模組
│   ├── external_replay.py           # 外部 replay 主要入口（從 external_replay/ 導入）
│   ├── external_replay/             # f1-race-replay 核心代碼
│   │   ├── arcade_replay.py         # ✅ 最新版本 - 可調整窗口、天氣數據
│   │   ├── f1_data.py               # ✅ 最新版本 - 遙測數據處理
│   │   └── lib/
│   │       └── tyres.py             # 輪胎數據工具
│   └── replay_wrapper.py            # 整合包裝器
└── main.py                           # 統一入口點
```

## 🎯 功能特色

### 來自 f1-race-replay 的功能：
- ✅ **歷史比賽回放** - 觀看過去的 F1 比賽
- ✅ **可調整窗口** - 支持窗口大小調整
- ✅ **互動式排行榜** - 點擊車手查看詳細遙測
- ✅ **天氣數據顯示** - 賽道溫度、空氣溫度、濕度、風向風速、降雨狀況
- ✅ **輪胎策略** - 實時顯示輪胎配方
- ✅ **賽道狀態** - 黃旗、紅旗、安全車、虛擬安全車
- ✅ **播放控制** - 暫停、快進、倒退、調速（0.5x - 4x）
- ✅ **車手遙測** - 速度、檔位、DRS 狀態

### 您專案獨有的功能：
- 🔮 **未來比賽預測** - 使用機器學習預測比賽結果
- 🤖 **AI 聊天助手** - 即時 F1 問答
- 📊 **ML 趨勢分析** - 車手表現趨勢預測

## 🚀 使用方法

### 1. 互動模式（推薦）

```bash
python main.py
```

選擇選項：
1. 🔮 預測未來比賽 (使用 ML 模型)
2. 📼 回放歷史比賽 (使用 f1-race-replay)
3. 📅 查看賽程表
4. ❌ 離開

### 2. 命令行模式

#### 回放歷史比賽（使用 f1-race-replay）
```bash
# 使用大獎賽名稱
python main.py --replay --year 2024 --gp Monaco

# 使用回合數
python main.py --replay --year 2024 --round 7

# 回放 Sprint 賽
python main.py --replay --year 2024 --gp Brazil --sprint
```

#### 預測未來比賽（使用 ML）
```bash
python main.py --predict --year 2025 --gp Monaco
python main.py --predict --year 2025 --gp Silverstone --speed 2.0
```

#### 查看賽程
```bash
python main.py --schedule
```

## 🎮 回放控制

在回放窗口中：

| 按鍵 | 功能 |
|------|------|
| `SPACE` | 暫停/繼續 |
| `←` / `→` | 快退 / 快進 |
| `↑` / `↓` | 增加/減少播放速度 |
| `1` | 設定速度 0.5x |
| `2` | 設定速度 1.0x |
| `3` | 設定速度 2.0x |
| `4` | 設定速度 4.0x |
| `R` | 重新開始 |
| **滑鼠點擊** | 點擊排行榜上的車手查看遙測 |

## 📊 顯示信息

### 主要 HUD（左上角）
- 當前圈數 / 總圈數
- 比賽時間
- 賽道狀態（綠旗/黃旗/紅旗/安全車/虛擬安全車）

### 天氣面板（左上角下方）
- 🌡️ 賽道溫度
- 🌡️ 空氣溫度
- 💧 濕度
- 🌬️ 風速和風向
- 🌧️ 降雨狀況

### 排行榜（右上角）
- 即時排名
- 車手代碼
- 當前輪胎配方圖標
- OUT 標記（退賽車手）

### 車手詳細信息（點擊排行榜後顯示）
- 速度（km/h）
- 當前檔位
- DRS 狀態（Off / Eligible / On）
- 當前圈數

## 🔧 技術細節

### 數據緩存
- 遙測數據會自動緩存到 `computed_data/` 目錄
- FastF1 數據緩存在 `.fastf1-cache/` 目錄
- 使用 `--refresh-data` 強制重新計算數據

### 性能優化
- 使用 NumPy 向量化運算
- 預計算賽道幾何數據
- 批量渲染優化
- 幀率：25 FPS（播放）

### 支持的賽季
- 2018 年至今的所有 F1 賽季
- 包括正賽和 Sprint 賽

## 🆕 最新更新（已整合）

從 f1-race-replay 最新版本整合的功能：
- ✅ 窗口可調整大小（Resizable window）
- ✅ 改進的排行榜準確性（基於賽道位置投影）
- ✅ 天氣數據顯示
- ✅ 滑鼠互動選擇車手
- ✅ 車手遙測詳細顯示
- ✅ 更好的 UI 佈局（避免與賽道重疊）

## 🐛 已知問題

### 來自上游 f1-race-replay 的已知問題：
1. 比賽開始前幾個彎道的排行榜可能不準確
2. 進站時排行榜會暫時受影響
3. 比賽結束時最終位置可能受影響

這些是遙測數據本身的限制，正在持續改進中。

## 📚 相關資源

- **上游項目**: [IAmTomShaw/f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay)
- **FastF1 文檔**: [FastF1 Documentation](https://docs.fastf1.dev/)
- **Arcade 文檔**: [Arcade Documentation](https://api.arcade.academy/)

## 🙏 致謝

- 回放系統核心代碼來自 [f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay) by Tom Shaw
- 數據來源: [FastF1](https://github.com/theOehrly/Fast-F1)
- 圖形引擎: [Python Arcade](https://api.arcade.academy/)

## 📝 授權

- 本專案的整合部分: MIT License
- f1-race-replay: MIT License
- 所有 F1 商標歸其各自所有者所有

---

**最後更新**: 2025-12-03
**整合版本**: f1-race-replay latest (2025-12-03)
