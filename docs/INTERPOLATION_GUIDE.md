# 幀插值設定指南 (Frame Interpolation Guide)

## 什麼是幀插值？(What is Frame Interpolation?)

幀插值會在原始 telemetry 數據點之間自動生成額外的幀，讓動畫更流暢。

**例如：**
- 原始數據：25 FPS (每秒25個數據點)
- `INTERPOLATION_FACTOR = 8`
- 生成幀數：25 × 8 = **200 FPS** (內部)
- 顯示幀率：60 FPS (受 VSync 限制)

## 當前設定 (Current Settings)

```python
# 位置：src/f1_data.py
INTERPOLATION_FACTOR = 8  # 生成 8 倍的幀數
```

### 這代表什麼？

| 設定 | 說明 |
|------|------|
| **原始採樣** | 25 FPS telemetry data |
| **插值倍數** | 8x |
| **內部幀數** | 200 FPS (25 × 8) |
| **每對點之間** | 7 個插值幀 |
| **顯示幀率** | 60 FPS (VSync) |

## 如何調整？(How to Adjust?)

### 方法一：修改設定檔

編輯 `src/f1_data.py` 第 22-26 行：

```python
# 找到這行
INTERPOLATION_FACTOR = 8

# 改成你想要的值
INTERPOLATION_FACTOR = 4   # 較少幀數，較快載入
INTERPOLATION_FACTOR = 10  # 更多幀數，超級流暢
```

### 方法二：不同設定的效果

| Factor | 內部 FPS | 插值幀 | 流暢度 | 記憶體 | 載入時間 | 建議用途 |
|--------|---------|--------|--------|--------|---------|---------|
| **2** | 50 | 1 | ⭐⭐ | 低 | 快 | 測試/舊電腦 |
| **4** | 100 | 3 | ⭐⭐⭐ | 中 | 中 | 標準使用 |
| **6** | 150 | 5 | ⭐⭐⭐⭐ | 中高 | 中 | 推薦設定 |
| **8** | 200 | 7 | ⭐⭐⭐⭐⭐ | 高 | 較慢 | 最佳體驗 ✨ |
| **10** | 250 | 9 | ⭐⭐⭐⭐⭐ | 很高 | 慢 | 極致流暢 |

### 當前推薦：**8** ✨

- 超級流暢的動畫
- 記憶體使用合理
- 載入時間可接受
- 最佳視覺體驗

## 技術細節 (Technical Details)

### 插值如何運作？

```python
# 在 src/f1_data.py 中
interpolated_dt = DT / INTERPOLATION_FACTOR  # 計算新的時間步長
timeline = np.arange(global_t_min, global_t_max, interpolated_dt)

# 對每個欄位進行線性插值
driver_data_array[:, driver_idx, FIELD_X] = np.interp(timeline, t_sorted, data["x"][order])
driver_data_array[:, driver_idx, FIELD_Y] = np.interp(timeline, t_sorted, data["y"][order])
# ... 其他欄位
```

### 插值的欄位

所有這些數據都會被插值：
- ✅ **X, Y 位置** - 車輛在賽道上的座標
- ✅ **距離** - 行駛距離
- ✅ **圈數** - 當前圈數
- ✅ **輪胎** - 輪胎類型
- ✅ **速度** - 車速 (km/h)
- ✅ **檔位** - 當前檔位
- ✅ **DRS** - DRS 狀態

### 再加上實時插值！

除了預生成的幀，`arcade_replay.py` 還會在**播放時**進行額外的插值：

```python
# 在 arcade_replay.py 中
interpolation = self.frame_index - int(self.frame_index)  # 0.0 到 1.0
x = current_x + (next_x - current_x) * interpolation
y = current_y + (next_y - current_y) * interpolation
```

**雙重插值系統：**
1. **預生成插值** - 數據載入時 (INTERPOLATION_FACTOR = 8)
2. **實時插值** - 播放時 (sub-frame interpolation)

**結果：** 超級絲滑的動畫！ 🚀

## 效能影響 (Performance Impact)

### 記憶體使用

```
原始數據大小 × INTERPOLATION_FACTOR = 總記憶體

例如：
- Factor 4: 100 MB → 400 MB
- Factor 8: 100 MB → 800 MB  ← 當前設定
- Factor 10: 100 MB → 1000 MB
```

### 載入時間

```
原始載入時間 × (1 + 0.2 × INTERPOLATION_FACTOR)

例如（估計）：
- Factor 4: 30 秒 → 54 秒
- Factor 8: 30 秒 → 78 秒  ← 當前設定
- Factor 10: 30 秒 → 90 秒
```

### 播放效能

✅ **不受影響** - 因為使用了批次渲染和 NumPy 優化
- 60 FPS 穩定
- CPU 使用率低

## 何時該調整？(When to Adjust?)

### 降低設定 (2-4) 如果：
- ⚠️ 電腦記憶體不足 (< 8 GB)
- ⚠️ 載入時間太長 (> 2 分鐘)
- ⚠️ 想快速測試

### 提高設定 (10+) 如果：
- ✨ 想要極致流暢的動畫
- ✨ 電腦配備好 (16+ GB RAM)
- ✨ 用於展示/錄製影片
- ✨ 不在意載入時間

## 實際範例 (Examples)

### 保守設定 (快速測試)
```python
INTERPOLATION_FACTOR = 2  # 50 FPS, 快速載入
```

### 標準設定 (日常使用)
```python
INTERPOLATION_FACTOR = 4  # 100 FPS, 平衡
```

### 推薦設定 (最佳體驗)
```python
INTERPOLATION_FACTOR = 8  # 200 FPS, 超級流暢 ✨
```

### 極致設定 (完美主義者)
```python
INTERPOLATION_FACTOR = 10  # 250 FPS, 極致流暢
```

## 如何驗證效果？(How to Test?)

```bash
# 1. 修改 src/f1_data.py 的 INTERPOLATION_FACTOR

# 2. 刪除快取（強制重新生成）
rm -r computed_data/*.npz

# 3. 執行測試
python main.py --replay --year 2024 --gp Monaco

# 4. 觀察：
# - 載入時間是否增加？
# - 動畫是否更流暢？
# - FPS 是否穩定在 60？
```

## 常見問題 (FAQ)

### Q: 為什麼要生成這麼多幀？
**A:** 因為原始 telemetry 只有 25 FPS，直接播放會很抖。插值後可以達到 200 FPS 內部幀率，再配合 60 FPS 顯示，就超級流暢了！

### Q: 會影響播放效能嗎？
**A:** 不會！因為使用了 NumPy 陣列和批次渲染，即使是 200 FPS 的數據也能流暢播放。

### Q: 改了之後要重新載入嗎？
**A:** 是的，需要刪除 `computed_data/` 資料夾中的快取檔案，讓系統重新生成。

### Q: 推薦設定是多少？
**A:** **8** 是最佳平衡！流暢度極高，記憶體使用合理，載入時間可接受。

### Q: 可以動態調整嗎？
**A:** 目前需要修改程式碼。未來可以考慮加入命令列參數 `--interpolation 8`。

## 視覺化比較

```
INTERPOLATION_FACTOR = 2
原始: ●-------●-------●-------●
插值: ●---○---●---○---●---○---●
      50 FPS (基本流暢)

INTERPOLATION_FACTOR = 4
原始: ●---------------●---------------●
插值: ●---○-○-○---●---○-○-○---●---○-○-○---●
      100 FPS (流暢)

INTERPOLATION_FACTOR = 8 ← 當前設定
原始: ●-------------------------------●
插值: ●-○○○○○○○-●-○○○○○○○-●-○○○○○○○-●
      200 FPS (超級流暢) ✨

INTERPOLATION_FACTOR = 10
原始: ●---------------------------------------●
插值: ●-○○○○○○○○○-●-○○○○○○○○○-●-○○○○○○○○○-●
      250 FPS (極致流暢)
```

## 總結

當前設定 `INTERPOLATION_FACTOR = 8` 提供：
- ✅ 超級流暢的動畫 (200 內部 FPS)
- ✅ 每對點之間 7 個插值幀
- ✅ 配合實時插值的雙重流暢系統
- ✅ 最佳的視覺體驗

**享受絲滑的 F1 賽事重播吧！** 🏎️💨

---

**修改後記得：**
1. 刪除 `computed_data/` 快取
2. 重新執行程式
3. 觀察流暢度提升！
