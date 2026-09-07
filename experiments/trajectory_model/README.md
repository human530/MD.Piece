# 疾病軌跡模型（模擬與分析流程）

依 `疾病軌跡模型_建置提示詞_v1.md`、`規格決定書_v1.md`、`規格決定書_v2_裁決.md`（2026-08-17/18）實作。
檢驗 H1（風險層內多種軌跡型態）、H2（軌跡資訊改變的是對象與時機而非排序）、H3（預警的型態依賴）。

## 執行

```bash
cd experiments/trajectory_model
python -m pytest tests -q                 # 20 個單元測試（FADE 等價、洩漏防護、β=0 無增益、分層混合、漂移起始、雙準則群數…）
python run.py --quick                     # 煙霧測試（n=400、1 seed；--quick --n 1500 可放大）
python -u run.py --jobs 12 --out results  # 完整格點：n=3000、5 seed、各變體（數小時）
python run.py --figures-only --out results   # 只由 results.json 重繪六張圖
```

輸出：`results/results.json`（所有數值結果：每個參數組合、每 seed、聚合平均／範圍、校準紀錄、τ 掃描、
洩漏警訊清單、figure_data）、`results/figures/fig1–7*.png`（黑白灰）、`results/assumptions.md`（假設／校準參數清單，供計畫書附錄）。

## 檔案

| 檔案 | 模組 | 內容 |
|---|---|---|
| `params.json` | 零 | 外部參數檔：每列附 `source`；`derived_from` 為 `assumption`／`calibrated_*` 者啟動時列印並寫入 assumptions.md |
| `fade_components.py` | — | 自 FADE `fade_sim.py` 逐字複製的元件（v1 §10）；僅 `simulate_S0` 加 `tau` 與外給 `mu_path` |
| `cohort.py` | 一 | 生成器（甲：OU 有色雜訊線性型；乙：雙穩態 SDE 於標準座標 y，平移到個案基線）、風險驅動事件、漂移起始日、校準（Δμ、λ0）、分層混合檢核、型別可分辨度 |
| `risk.py` | 二 | 基線 logistic 風險分數（pilot 世代配適一次）、五／十分位分層 |
| `clustering.py` | 三 | 方法 A（正交多項式係數 → GMM）／B（特徵 → k-means，分類似然 BIC）；BIC＋bootstrap 穩定度選 K；時間打亂置換檢定；對生成器標籤 ARI/NMI |
| `prediction.py` | 四 | 靜態 vs 地標動態；C 指數／Brier；淨重新分類（絕對＋相對閾值）；介入時點位移（每 90 天）|
| `warning.py` | 五 | 滾動 AR(1)/SD（= FADE 定義，向量化）；逐前綴 Kendall τ；區塊置換共同虛無＋同質性；單尾上升警報＋下降型偏離計數；三種提前期；趨勢外推基準（burn-in）|
| `figures.py` | — | 六張圖（含 v2 壹 的 τ 掃描圖）|
| `run.py` | — | 單一入口；變體與格點；unreachable 留白；分層混合檢核停止；平行；聚合；洩漏警訊 |
| `tests/` | — | 每個模組至少一個測試，每個測試註明它守住哪條方法學規則 |

## 生成器（v2 後）

- **共同刻度**：序列單位為 eGFR（低＝差）。兩型 x₀ 自同一基線分布抽（KDIGO G2–G4 均勻）；型別由潛在易感度決定（`type_link_beta`），不得由 x₀ 決定（v2 捌）。
- **翻轉型**：雙穩態 SDE 在標準座標 y 上跑（FADE `simulate_S0`，τ=60），以固定尺度 b（健康態↔90、摺疊點↔60 的參考幾何）平移到個案基線 eGFR = eGFR₀ − b·(y − y_L)，因此翻轉形狀與 CSD 可偵測性與 x₀ 無關。每人抽漂移起始日（0–1095 天）與漂移期（90–730 天），起始日前 μ 固定、序列定態（v2 玖）。
- **線性型**：eGFR = eGFR₀ − 斜率·t + OU 雜訊；斜率子類別取 O'Hare 2012。
- **三個時點**：`t_crit`（μ 跨 μc）、`t_threshold`（eGFR 首次 ≤ 15，描述性里程碑）、`t_event`（風險驅動：λ(t)=λ₀·exp(β·(15−max(eGFR,0))/10)，β 固定 ln 3（KFRE 量級假設），λ₀ 校準到五年事件率 22.5%）（v2 參）。
- **校準**：Δμ 中位以 median(t_threshold − t_crit)=180 天反推（參考幾何 x₀=90）；不可達的 τ／時程格標 `unreachable` 留白（v2 壹）。

## 方法章節要寫進去的三件事（v2 伍、柒、拾壹）

1. **時間打亂置換的虛無假設**是「該序列的時間順序不帶資訊」，**不是**「該序列無變異」：置換保留每條序列的水準與總變異，因此顯著代表變化的**形狀**有意義，而非幅度有意義。
2. **方法學自我檢核實例**（單元測試抓出、已修）：(a) 常見的 k-means 等向 BIC 在純高斯雜訊上會一路選到 k_max → 改用分割的分類似然 BIC（完整共變異數），雜訊上選 K=1、植入兩群選 K=2；(b) 原始多項式基底 1, t, t² 高度共線，z-score 不解決共線，純雜訊的高次項會被放大而沿雜訊維度切群 → 改用正交（Legendre）基底、不做 z-score。這兩項說明為何以兩種演算法互相對照。
3. **限制**：方法 A 是「每人 OLS 係數 → GMM」的兩階段近似，非逐點似然的完整群組軌跡模型；已知偏誤方向——忽略係數的估計不確定性，短序列／高雜訊時類內散布被高估、類界模糊、傾向低估類別分離度而以較多小群補償（本研究序列長 1825 點，此偏誤較小，但仍應報告 `hit_kmax`）。

## v2 裁決落地對照

| v2 | 落地 |
|---|---|
| 壹 τ | 主分析 τ=60、窗 {21,42,60,90}；τ 14/30 與 12 個月目標 → `unreachable` 留白；fig6 τ 掃描曲線，圖說載明實測值（τ=30 最長 94 天、τ=14 最長 44 天）|
| 貳 起始水準 | 主分析 `kdigo_g2_g4_uniform`（`assumption_alignment_corrected`）；`ohare_ranges` 示範變體實際跑（分層單一型別 1.00、定態期型別可分辨 AUC 0.96——即對齊偏誤的樣子）|
| 參 結局 | 風險驅動；三時點；β 固定 HR 3（C 對 β 非單調，見 params）；`test_hazard_beta_zero_makes_trajectory_uninformative` |
| 肆 警報 | 主警報單尾上升（τ 同為正且 > 95 分位）；下降型偏離獨立計數（`downward`）；假設檢定雙尾 |
| 伍 置換 | 方法說明（上）＋程式註解 |
| 陸 burn-in | 趨勢基準 90 天內不警報、自偽警報分母扣除（`alarms_from_flags(start_day)`）|
| 柒 方法缺陷 | 寫入方法章節（上）|
| 捌 起始重疊 | 同分布 x₀、型別由 s 決定、分層單一型別 >90% 停止（pilot 與每 job）|
| 玖 漂移起始 | `build_mu_path`；並報 t_event−t_crit 與 t_event−首次警報；>1 年標為設定產物 |
| 拾 不宣稱 | 計畫書表二已加 N7（絕對提前期）、N8（C 增益絕對大小）|
| 拾壹 | 假設清單 → `results/assumptions.md`（含三道閘門）；圖 1／5 由本輪重跑產生 |
| 附錄壹 群數 | k_max=8；報告 K = 同時滿足「K→K+1 BIC 改善 < 整段 5%」且「穩定度 ≥ 0.60（Hennig）」的最小 K；輸出 BIC／穩定度曲線（fig7 雙軸）、`k_ceiling_hit`（BIC 到 8 仍單調改善）、`continuous_heterogeneity`（撞頂且穩定度遞減 → 該層異質性連續、離散劃分不受支持）、`n_true_labels`／`over_split_by` |
| 附錄貳 線性定態前期 | 線性型下降起始日與翻轉型漂移起始日同分布（U(0,1095)）；報兩型定態期分布與 KS p；`linear_from_day0` 敏感度變體 |
| 附錄參 閘門 | 跑格點前：分層單一型別 ≤ 0.9；β=0 時 \|ΔC\| < 0.01（n=12000 檢核世代、三地標帶號平均——n=1500 的 ΔC 雜訊 ±0.02 會誤判）；定態期型別 AUROC 0.45–0.55；任一未過即中止並寫 gates.json |

## 變體

`default`（τ60）、`fliptime90`、`linear_from_day0`、`ohare_ranges`（示範）、`dropout`（模組四）、`beta_x0.5`／`beta_x2`（模組四）；
`tau14`／`tau30`／`fliptime365` → unreachable 留白（僅入 fig6）。

## 誠實提醒

合成資料上的相對比較；絕對數值不構成臨床主張。
