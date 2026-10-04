# F1 賽事預測模擬器 🏎️

[English](README.md) | **繁體中文**

用 Python 打造的 F1 賽事「預測 + 模擬 + 策略分析」工具。
它會根據車隊實力、車手積分與賽道特性，預測一場大獎賽的排位，
接著用 [Arcade](https://api.arcade.academy/) 開一個視窗，把整場比賽「演」給你看，
最後算出完賽成績、積分，還能畫出名次變化圖與輪胎策略圖。

> 📼 想看**真實的歷史比賽回放**？本專案有內建一個簡易版（`--replay`），
> 但更完整的體驗請用 [f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay)。

> ⚠️ **已知問題**：目前預測後選擇開啟模擬視窗會直接出錯（`AttributeError: DRIVERS_2025`），
> 詳見下方「[可以改進的地方](#可以改進的地方)」第 1 點。排位預測表格、策略分析和賽程不受影響。

---

## 目錄

- [這個專案能做什麼](#這個專案能做什麼)
- [運作原理（白話版）](#運作原理白話版)
- [安裝](#安裝)
- [使用方式](#使用方式)
- [輸出檔案](#輸出檔案)
- [執行測試](#執行測試)
- [專案結構](#專案結構)
- [常見問題](#常見問題)
- [可以改進的地方](#可以改進的地方)
- [致謝與授權](#致謝與授權)

---

## 這個專案能做什麼

| 功能 | 說明 | 需要網路？ |
|------|------|-----------|
| 🔮 賽事預測 | 預測排位前十名與「預測信心度」，並推薦進站策略 | 選用（沒網路時用內建資料） |
| 🎬 比賽模擬視窗 | 在賽道上即時模擬整場比賽，含進站、超車、安全車、退賽（⚠️ 目前有 bug 無法開啟，見已知問題） | 選用 |
| ⏱️ 時間軸拖曳 | 視窗底部進度條可點擊或拖曳跳到任何時刻，並標示黃旗/安全車/紅旗 | 否 |
| 🏁 完賽成績 | 最終排名、差距、退賽、世界冠軍積分（含最快圈加分），可匯出 JSON / CSV | 否 |
| 📊 比賽圖表 | 名次變化折線圖、輪胎策略圖，存成 PNG | 否 |
| 🎯 策略分析 | 一停 vs 兩停比較、Undercut / Overcut 判斷、油重對圈速影響 | 否 |
| 📅 賽程表 | 查看任一年度賽程（FastF1 取得，失敗時改用內建 2025 賽程並提示） | 選用 |
| 📼 歷史回放 | 用 FastF1 真實遙測資料回放過去的比賽 | **需要** |
| 🤖 AI 助理 | 在模擬視窗中問 F1 問題（使用 Groq API） | **需要** + API 金鑰 |

---

## 運作原理（白話版）

```
 ┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐    ┌──────────────┐
 │ 1. 取得資料   │ →  │ 2. 預測排位       │ →  │ 3. 逐圈模擬比賽    │ →  │ 4. 結算與輸出 │
 │ FastF1 賽程   │    │ 車隊實力 + 積分   │    │ 輪胎老化、進站、   │    │ 成績表、積分  │
 │ 車手名單      │    │ + 隨機變化        │    │ 超車、安全車、退賽 │    │ 圖表、匯出    │
 │ (失敗→內建)   │    │                  │    │                   │    │              │
 └──────────────┘    └──────────────────┘    └───────────────────┘    └──────────────┘
```

1. **取得資料**：`FutureRaceDataProvider` 取得賽程與車手。
   年份是 2025 時直接使用程式內建的 2025 資料；其他年份會先嘗試 FastF1，失敗才改用內建資料。
   目前只有「查看賽程表」會在畫面上提示「正在使用內建資料」，預測流程不會提示。
2. **預測排位**：依「車隊實力分數 + 車手積分 + 隨機變化」排序出發位置。
   因為有隨機成分，**每次執行結果都會不太一樣**。
3. **逐圈模擬**：`PredictedRaceSimulator` 搭配 `race_dynamics.py`
   模擬輪胎退化、進站損失時間、超車機率、DRS、安全車與退賽。
4. **結算輸出**：`race_results.py` 從模擬結果算出最終成績與積分；
   `dashboard/charts.py` 用 matplotlib 畫圖。

---

## 安裝

建議使用 **Python 3.10 以上**（CI 在 3.10 與 3.12 上測試）。

```bash
# 1. 下載專案
git clone https://github.com/yakiniku35/F1_strategies.git
cd F1_strategies

# 2.（建議）建立虛擬環境，避免弄亂系統的 Python
python -m venv .venv
source .venv/bin/activate        # Windows 請用：.venv\Scripts\activate

# 3. 安裝套件
pip install -r requirements.txt

# 4.（選用）啟用 AI 助理：到 https://console.groq.com 申請金鑰後
echo "GROQ_API_KEY=你的金鑰" > .env
```

> 💡 `requirements.txt` 包含 FastF1、scikit-learn、xgboost 等較大的套件，
> 第一次安裝會花一點時間。

---

## 使用方式

### 方法一：互動選單（最適合新手）

```bash
python main.py
```

會出現：

```
╔══════════════════════════════════════════════════════════╗
║           F1 Race Prediction Simulator 🏎️                ║
╠══════════════════════════════════════════════════════════╣
║  1. 🔮 預測未來比賽 (Predict Future Race)                ║
║  2. 📼 回放歷史比賽 (Replay Historical Race)             ║
║  3. 📅 查看賽程表 (View Schedule)                        ║
║  4. 🎯 策略分析 (Strategy Analysis)                      ║
║  5. ❌ 離開 (Exit)                                       ║
╚══════════════════════════════════════════════════════════╝
```

輸入數字後依提示輸入年份（2018–2030）和大獎賽名稱（例如 `Monaco`）即可。

### 方法二：命令列參數

```bash
# 預測 2025 年摩納哥大獎賽
python main.py --predict --year 2025 --gp Monaco

# 用輪次編號指定比賽，並跳過模型訓練（啟動比較快）
python main.py --predict --year 2025 --round 8 --no-train

# 以 2 倍速播放模擬
python main.py --predict --gp Silverstone --speed 2.0

# 查看賽程（不指定年份 = 今年）
python main.py --schedule
python main.py --schedule --year 2025

# 策略分析：銀石賽道、52 圈
python main.py --strategy --track Silverstone --laps 52

# 回放 2024 年摩納哥（需要網路，第一次會下載大量資料）
python main.py --replay --year 2024 --gp Monaco
```

### 參數一覽

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--predict` | 預測並模擬未來賽事 | — |
| `--replay` | 回放歷史賽事（需要網路） | — |
| `--schedule` | 查看賽程表 | — |
| `--strategy` | 策略分析 | — |
| `--year` | 年份 | 預測／賽程：今年；回放：2024 |
| `--gp` | 大獎賽名稱，例如 `Monaco`、`Silverstone` | — |
| `--round` | 輪次編號（可取代 `--gp`） | — |
| `--speed` | 模擬播放速度 | `1.0` |
| `--no-train` | 跳過 ML 模型訓練 | 關閉 |
| `--track` | 策略分析用的賽道名稱 | `Silverstone` |
| `--laps` | 策略分析用的總圈數 | `50` |

### 方法三：整合管線（進階）

```bash
python run_integrated.py --year 2024 --gp Monaco --mode full
```

`--mode` 可選 `full`、`predict-only`、`tables-only`、`simulation-only`，
輸出資料夾用 `--output` 指定（預設 `output/`）。

---

## 輸出檔案

| 檔案 / 資料夾 | 產生時機 |
|---------------|----------|
| `results_<年份>_<大獎賽>.json` / `.csv` | 模擬結束後選擇匯出成績 |
| `charts/<年份>_<大獎賽>/` | 模擬結束後選擇產生圖表（PNG） |
| `strategy_*.json` / `.csv` | 策略分析時選擇匯出 |
| `cache/`、`.fastf1-cache/` | FastF1 與 ML 模型快取（可放心刪除，會重新下載） |

以上都已列在 `.gitignore`，不會被誤推上 GitHub。

---

## 執行測試

```bash
pip install pytest
pytest
```

測試涵蓋純邏輯部分（成績與積分、輪胎退化、策略評分、賽程載入、退賽模擬、回放內插），
**完全不需要網路**。若沒安裝 FastF1、scikit-learn 等大型套件，
`tests/conftest.py` 會自動用替身模組代替。每次 push 時 GitHub Actions 也會自動跑同一套測試。

---

## 專案結構

```
F1_strategies/
├── main.py                     # 主程式入口（互動選單 + 命令列）
├── run_integrated.py           # 整合管線入口（進階）
├── requirements.txt            # 相依套件
├── src/
│   ├── simulation/
│   │   ├── future_race_data.py # 賽程、車手、車隊實力（含離線備援資料）
│   │   ├── race_simulator.py   # 預測排位 + 逐圈模擬比賽
│   │   ├── race_dynamics.py    # 輪胎退化、進站、超車、DRS
│   │   └── track_layouts.py    # 賽道形狀（FastF1 → 內建 → 橢圓備援）
│   ├── strategy_analyzer.py    # 進站策略分析
│   ├── race_results.py         # 完賽成績、積分、匯出
│   ├── dashboard/              # 圖表、表格、預測疊加層
│   ├── arcade_replay.py        # 預測模擬的 Arcade 視窗
│   ├── external_replay.py      # 歷史回放的 Arcade 視窗
│   ├── f1_data.py / external_f1_data.py  # FastF1 遙測資料處理
│   ├── ml_predictor.py         # 基本 ML 預測器
│   ├── ml_enhanced.py          # 增強版 ML（集成學習，目前只在 examples/ 使用）
│   ├── ai_chat.py              # Groq AI 助理
│   └── integration/pipeline.py # run_integrated.py 用的整合流程
├── tests/                      # pytest 測試
├── examples/                   # ML 模型範例腳本
├── docs/                       # 開發過程的技術文件
└── images/tyres/               # 輪胎圖示
```

---

## 常見問題

**Q：Linux 上安裝 Arcade 失敗？**
```bash
sudo apt-get install python3-dev libgl1-mesa-dev
pip install arcade
```

**Q：為什麼每次預測結果都不一樣？**
排位與比賽事件（超車、安全車、退賽）都含隨機成分，這是設計上的選擇，
讓你可以多跑幾次看看「各種可能的劇本」。

**Q：FastF1 資料怪怪的？**
刪掉快取資料夾重新下載即可：
```bash
rm -rf .fastf1-cache/ cache/
```

**Q：沒有網路可以用嗎？**
大部分可以。預測、策略分析、賽程都有內建備援資料；歷史回放與 AI 助理需要網路。
要注意 FastF1 仍是必裝套件（預測與模擬模組在載入時就會匯入它），只是不一定要連線。

---

## 可以改進的地方

以下是閱讀程式碼後整理出來的建議，依「影響大小」大致排序。

### 🔴 優先處理（影響正確性或使用者認知）

1. **模擬視窗會當掉（bug）**
   `generate_simulated_frames()` 會呼叫 `_get_team_colors()`，而它在
   `src/simulation/race_simulator.py` 第 691 行讀取 `self.data_provider.DRIVERS_2025`，
   但 `FutureRaceDataProvider` 已經沒有這個屬性（內建名單現在叫 `FALLBACK_DRIVERS`，實際名單由 `drivers` 屬性提供），
   所以會拋出 `AttributeError`。修法很小：把迴圈改成 `for driver in self.data_provider.get_drivers_list():`，
   並補一個呼叫 `generate_simulated_frames()` 的測試，避免再發生。

2. **ML 模型訓練了卻沒有被使用**
   `main.py` 的 `predict_future_race()` 會花幾分鐘訓練 `PreRacePredictor`，
   但訓練好的 `predictor` 之後完全沒被用到——排位其實是
   `FutureRaceDataProvider.estimate_qualifying()` 用「車隊實力 + 積分 + `random.uniform(-5, 5)`」算出來的。
   建議：把模型接進預測流程，或先預設不訓練，並在 README 誠實說明預測是「啟發式 + 隨機」。

3. **「預測信心度」不是真正的信心度**
   目前是依排名與車隊實力給的經驗值，並非模型輸出的機率。建議改名（例如「參考指數」）
   或改用 `ml_enhanced.py` 中已寫好的 `predict_with_confidence()`。

4. **結果無法重現**
   程式多處使用 `random.seed(int(time.time() * 1000))`，同樣的輸入每次結果都不同，
   也沒辦法除錯。建議加一個 `--seed` 參數，預設隨機、指定時可重現。

5. **英文 README 與實際程式不一致**
   英文版說「不提供歷史回放」，但 `main.py --replay` 其實存在；
   `--year` 寫「預設 2025」，實際是「今年」；舊版中文 README 還提到不存在的 `--refresh-data`。
   另外，預測流程沒有檢查 `schedule_source` / `drivers_source`，用到內建備援資料時不會告訴使用者。

### 🟡 結構整理（讓程式更好維護）

6. **重複的模組**：`f1_data.py` vs `external_f1_data.py`、`arcade_replay.py`（2000 行）vs
   `external_replay.py`（近 1000 行）功能高度重疊，可合併或抽出共用部分。
7. **`ml_enhanced.py` 沒接上主程式**，只在 `examples/` 裡用到。
8. **`temp_replay` 是失效的 git 子模組連結**（沒有 `.gitmodules`），CI 還得特別繞過，建議移除。
9. **`docs/` 有 17 份文件**，很多是開發紀錄（`*_COMPLETE.md`、`*_SUMMARY.md`），
   可以整理成一份 `CHANGELOG.md` 加幾份真正的使用指南；`docs/f1_tracl.txt` 檔名應是 `track` 的錯字。
10. **相依套件太重且沒分層**：`pytest` 被放進 `requirements.txt`；`groq`、`python-dotenv` 標示「選用」
   但 `arcade_replay.py → ai_chat.py` 在最上層就 `import dotenv`，沒裝會直接當掉。
   建議拆成 `requirements.txt`（核心）、`requirements-ml.txt`、`requirements-dev.txt`，
   或改用 `pyproject.toml` 的 optional dependencies。
11. **小地方**：`interactive_mode()` 用遞迴回到主選單（應改成 `while` 迴圈）；
    多處 `except Exception as e:` 卻沒用到 `e`，錯誤原因被吞掉；
    年份範圍 `2018–2030` 寫死在程式裡。

### 🟢 功能擴充：做成輕量網頁版

目前所有功能都要在本機裝 Python + Arcade（需要 OpenGL 視窗），沒辦法直接放上網。
好消息是：**策略分析（`strategy_analyzer.py`）和成績計算（`race_results.py`）只用到 Python 標準函式庫，
跟畫面完全分開**，現在就能搬上網頁。

不過**完整的比賽模擬還不行**：`race_simulator.py` 會匯入 `track_layouts.py` 和 `f1_data.py`，
這兩個檔案一載入就 `import fastf1`；`src/simulation/__init__.py` 也會連帶載入它們，
所以連 `race_dynamics.py` 都沒辦法單獨匯入而不碰到 FastF1。
要把模擬搬上網頁，得先把 FastF1 資料層拆開（例如改成需要時才在函式裡 `import fastf1`）。

依「輕量程度」由高到低有三種做法：

| 做法 | 說明 | 優點 | 缺點 |
|------|------|------|------|
| **A. 純靜態網頁（最輕量，推薦）** | 用 Python 預先跑好賽程、預測結果、策略比較，輸出成 JSON；網頁用原生 HTML + JavaScript + `<canvas>` 讀取並播放 | 免伺服器、免費放在 **GitHub Pages**、載入快 | 無法即時重新計算（可用 GitHub Actions 每週自動更新） |
| **B. 瀏覽器內跑 Python（Pyodide）** | 用 [Pyodide](https://pyodide.org/) 直接在瀏覽器執行 `strategy_analyzer.py`、`race_results.py`（模擬要先解耦 FastF1） | 仍是靜態網頁，但可即時互動計算 | 首次載入約 10 MB 以上；FastF1 無法在瀏覽器執行 |
| **C. 小型 API 伺服器** | 用 FastAPI / Flask 包一層 API，前端呼叫 | 功能最完整，可即時抓 FastF1 | 需要租伺服器（Render、Fly.io 等），較不輕量 |

建議的第一步是 **A**：
1. 新增一個 `export_web_data.py`，把 `view_schedule`、預測排位、`generate_simulated_frames()`
   （降採樣到每圈幾個點）、策略比較輸出成 `web/data/*.json`。
   這一步在本機或 GitHub Actions 跑，可以照常使用 FastF1，但要先修好上面第 1 點的 bug。
2. 在 `web/` 放一個 `index.html`（不使用任何框架），用 `<canvas>` 畫賽道與車子、用表格顯示成績。
3. 開啟 GitHub Pages，指向 `web/` 資料夾。
4. （選用）設定 GitHub Actions 每週重新產生 JSON，讓資料保持最新。

---

## 致謝與授權

- 歷史回放靈感與部分程式：[f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay)（Tom Shaw）
- 資料來源：[FastF1](https://github.com/theOehrly/Fast-F1)
- 圖形引擎：[Arcade](https://api.arcade.academy/)

Formula 1 及相關商標為其各自所有人之財產。本專案僅供學習與教育用途。

授權條款：[MIT License](LICENSE)
