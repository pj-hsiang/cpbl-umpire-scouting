# 研究方法與解讀限制｜Methodology

本文件說明《今天誰主審？》資料處理、建模與歷史比較的基本邏輯。完整實作請以目前 Repository 內的 `cpbl_umpire_pipeline.py` 為準。

## 1. Dataset

資料來源為第三方 [CPBL Umpire Scorecard](https://github.com/yuetsu001/cpbl-umpire-scorecard)，分析快照包含 36,245 筆逐球紀錄、239 場、18 位主審。資料來源可能更新；新資料跑出的結果不一定與舊版相同。

關鍵欄位：`pitch_x`、`pitch_z`、`pitch_sz_top`、`pitch_sz_bottom`、`pitch_called`、`pitch_true_call`、`pitch_dist_cm`、`pitch_count_b`、`pitch_count_s`、`umpire_name`、`game_id`、`date`。

## 2. Source-defined strike-zone geometry

對每顆球，定義 `x` 與 `z` 為球點座標，`b`、`t` 為該球的下、上界。資料來源推定的核心矩形水平半寬為 0.22m，擴張半徑 0.0369m。

```python
import numpy as np

dx = np.maximum(np.abs(x) - 0.22, 0)
dz = np.maximum.reduce([b-z, z-t, np.zeros(len(z))])
outside = np.hypot(dx, dz)
inside = (np.abs(x) <= 0.22) & (z >= b) & (z <= t)
inside_depth = np.minimum.reduce([0.22-np.abs(x), z-b, t-z])
signed = np.where(inside, -inside_depth, outside)
predicted_true_call = np.where(signed <= 0.0369, 'STRIKE', 'BALL')
predicted_distance_cm = np.abs(signed - 0.0369) * 100
```

在既有 36,245 筆快照中，這套公式重現了來源的理論判決與邊界距離（僅剩浮點誤差）。這是對**第三方資料演算法**的驗證，而不是對官方好球帶的認證。

## 3. Model A / B / C

- A：投球水平位置、相對上下界的標準化高度、位置二次項、球數。
- B：A + 主審身分。
- C：B + 主審 × 位置交互項。

主要採 Logistic Regression 與 L2 正則化；依時間切分訓練／測試，避免把同一時期全部資料當成樣本外證據。以 Log Loss、Brier Score、AUC 評估。

樣本外表現改善不等於主審偏好「顯著成立」；需要看改善幅度、跨場次不確定性與時間穩定性。

## 4. Historical comparison

對距離來源理論好球帶邊界 10cm 以內的投球，分析主審的實際判決與基準模型預期之差，再與同一區域其他主審的歷史平均殘差比較。

**相對差異（百分點） = 100 × [該主審的平均(實際好球指標 − 預期機率) − 其他主審的相同平均值]。**

這不是嚴格控制所有投球條件的因果估計；同區域內的球點分布仍可能不同。

## 5. League rankings

支援五區：High、Low、Left、Right、Corner。可查看：

- 邊緣球的實際好球率。
- 在「來源理論壞球」條件下，被主審判為好球的比例。
- 在「來源理論好球」條件下，被主審判為壞球的比例。

後兩者的**分母不同**；不能拿來直接等同主審整體誤判率。尤其在指定球數、指定區域時，要先檢查分母球數與涵蓋場次。

## 6. Uncertainty and sample warning

以比賽為群集單位的 Bootstrap 對歷史差異估計不確定性；此處不一定完整納入基準模型重新估計的誤差。

目前介面**不提供 A–D 等級**。`E` 只代表相應統計格的樣本不足，不是主審能力、判決品質或公平性的負面評比。請同時查看球數、場次、研究期間與資料來源限制。

## 7. Interpretation

這是供球迷研究、探索的歷史資料工具。請勿將「理論判決」當成官方判決正確答案，也不要把歷史觀察結果作為未來單場結果的保證。

