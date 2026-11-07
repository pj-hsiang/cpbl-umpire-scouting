# 今天誰主審？｜CPBL Umpire Scouting Dashboard

**An interactive baseball analytics project exploring historical strike-zone tendencies among CPBL home-plate umpires.**

以中華職棒（CPBL）逐球判決資料為基礎，建立主審歷史好球帶情蒐平台。透過 Python、機器學習與統計分析，讓使用者比較主審在不同球數、好球帶區域的判決傾向，並查看樣本數與資料限制。

> **[開啟互動式網站（Live Demo）](https://YOUR_GITHUB_USERNAME.github.io/cpbl-umpire-scouting/)**  
> 上線後請將 `YOUR_GITHUB_USERNAME` 替換成你的 GitHub 帳號；如果 Repository 不是 `cpbl-umpire-scouting`，也需修改網址。

## Project at a glance

| 項目 | 內容 |
|---|---|
| Dataset | 36,245 筆逐球判決紀錄 |
| Coverage | 239 場比賽、18 位主審（本次分析資料快照） |
| Focus | 好球帶幾何驗證、主審判決差異、歷史情蒐與視覺化 |
| Methods | Logistic Regression、時間樣本外驗證、Game-cluster Bootstrap |
| Tech stack | Python、Pandas、NumPy、SciPy、scikit-learn、HTML/CSS/JavaScript |
| Deployment | GitHub Pages（單一 HTML，內嵌統計資料） |

## Why this project?

球迷常討論不同主審是否有自己的好球帶，但直覺觀察容易受少數爭議判決影響。本專案嘗試以完整歷史逐球紀錄，區分「看起來不同」與「在球點、球數等條件下仍存在的歷史判決差異」，並把分析轉換成一般球迷能理解的互動介面。

這是一個**歷史資料探索工具**，不是官方誤判評鑑，也不保證下一場比賽的判決結果。

## Features

- **主審情蒐卡**：查詢 18 位主審的歷史五區判決特徵。
- **雙主審比較**：在同一區域比較不同主審的歷史傾向。
- **球數篩選**：預設 `0 壞 0 好`，可切換不同球數（包含 `2 壞 2 好`），也可查看所有球數彙總。
- **聯盟排行榜**：依上、下、座標左、座標右與角落，排序歷史好球率及理論好／壞球與實際判決不一致的比例。
- **樣本不足提示**：`E` 僅表示該統計格的投球數或場次不足，**不代表主審能力或判決品質**。
- **手機與桌面瀏覽**：透過 GitHub Pages 分享互動網站；資料已內嵌於 HTML。

## Research workflow

```text
Raw pitch-call CSV
       ↓
Data cleaning & validation
       ↓
Reverse-engineer the source-defined strike-zone geometry
       ↓
Verify theoretical calls and boundary distances
       ↓
Train and compare Model A / B / C on a chronological holdout
       ↓
Describe umpire/region/count tendencies & uncertainty
       ↓
Build a standalone interactive HTML dashboard
       ↓
Deploy with GitHub Pages
```

### Strike-zone geometry

本專案曾以來源資料反推並驗證其**理論好球帶演算法**：矩形核心水平半寬約 `0.22 m`，搭配圓角擴張參數約 `0.0369 m`。資料快照中的 36,245 筆理論判決及邊界距離可由該公式還原。

**這是第三方資料來源的計算定義，不等同於中職官方規範；也不應將圓角參數直接當作官方球半徑。**

### Machine learning validation

以球點位置與球數為基本資訊，對照三種 Logistic Regression 設定：

| Model | 特徵 | 研究目的 |
|---|---|---|
| A | 球點位置、球數 | 不含主審身分的基準模型 |
| B | A + 主審身分 | 檢查主審資訊是否帶來額外預測價值 |
| C | B + 主審 × 球點交互項 | 探索不同主審是否有區域性差異 |

使用**依日期切分的訓練集與測試集**，比較 Log Loss、Brier Score 與 AUC。先前的研究實驗中，Model B 的樣本外 Log Loss 略低於 Model A，但改善幅度有限；這是預測訊號，不代表「已證明每位主審有固定偏好」。完整數值請以重新執行後的報表為準。

更完整的方法、解讀限制和再現方式見 [研究方法說明](docs/METHODOLOGY.md)。

## Getting started

### Just want to use the dashboard?

點選上方 **Live Demo** 即可，不需安裝 Python。網站是單一 `index.html`，統計結果已內嵌，瀏覽器無須另外下載 CSV。

### Reproduce the analysis

準備來源資料 `umpire_pitch_calls.csv`，置於專案根目錄，然後執行：

```bash
python3 -m pip install -r requirements.txt
python3 cpbl_umpire_pipeline.py --input umpire_pitch_calls.csv --out output --bootstrap 4000
```

分析結果會輸出到 `output/`；正式網頁為 `output/umpire_scouting.html`。將它複製並命名為專案根目錄的 `index.html`，即可部署至 GitHub Pages。

> 資料下載與公開分享需依第三方資料來源的使用條件進行。本 Repository 建議只發布程式碼、文件與已確認可公開的統計網站；原始 CSV 不必公開。

## Repository structure

```text
cpbl-umpire-scouting/
├── index.html                  # GitHub Pages 網站首頁
├── README.md                   # 專案介紹
├── cpbl_umpire_pipeline.py     # 一鍵分析流程
├── requirements.txt            # Python 套件
├── .gitignore                  # 不公開的原始/暫存資料排除規則
└── docs/
    └── METHODOLOGY.md         # 分析方法與研究限制
```

## Data source & limitations

- 資料來源：[CPBL Umpire Scorecard（第三方社群專案）](https://github.com/yuetsu001/cpbl-umpire-scorecard)。本專案與中華職棒官方無關。
- 本資料中的 `pitch_true_call` 是**來源定義的理論判決**，不能直接視為官方判決對錯。
- `Left` / `Right` 表示原始座標方向，未必對應打者視角的內角與外角。
- 歷史好球率會受球點分布、球數和樣本量影響；比較結果屬觀察性分析，非因果結論。
- 指定球數、指定區域的資料可能稀疏，因此介面提供樣本不足警示。
- 部署 HTML 內嵌的統計資料可由訪客讀取，公開前請確認內容與轉載條件。

## Future work

- 更細緻的二維好球帶熱圖與不確定性呈現。
- 更完整的時間滾動驗證與小樣本收縮估計。
- 當日主審名單確認後，探索賽前／賽中判決機率輔助功能（**尚未實作**）。

---

**Project type:** Personal baseball analytics / data science portfolio project.  
**Disclaimer:** 本工具旨在協助理解歷史資料，不評定主審好壞，也不提供官方判決裁定。
