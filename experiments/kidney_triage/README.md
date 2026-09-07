# 腎損傷分流模型（Stage 0–5）

依《建置指示 v11》（2026-08-28）建置。**在醫師開出任何專項檢驗之前，用已經有的常規檢驗把範圍縮小，並指出下一步該開哪一張單。**
不是從抽血推測病名——Stage 6（確認：培養／抗體／基因／切片）不在模型內，由醫師執行。

```bash
cd experiments/kidney_triage
python pipeline.py              # 完整管線（含所有 assert）→ results/*.json、feature_importance.csv
python make_report.py           # → results/report.md、results/limitations.md
python figures.py               # → results/figs/figA–figE（PNG 300dpi ＋ PDF）
python ui/export_viz.py         # → ui/index.html（單一離線頁，含個案分流試算）
python pipeline.py --quick      # 小樣本冒煙測試（n=800）
```

## ⚠ 本次建置無真實資料

世代由 `src/datagen.py` 依「資料契約」的欄位規格合成，而**生成器的全部參數（盛行率、通道效應、可得比例、開立行為）
無文獻依據**——`params/params.json` 的 `generator` 列 `value: null`，以 placeholder 佔位執行，啟動時列印警告（指示 一之七）。
**所有效能數字都是這些假設的產物，僅示範管線與方法學，不構成對任何真實族群的證據。**
要接真實資料：把 `datagen.simulate()` 換成讀檔，欄位對齊第二節資料契約即可，其餘不動。

## 七條硬性約束的落地位置

| 約束 | 實作 | 執行期斷言 |
|---|---|---|
| 1 通道消耗規則（封存集不得為特徵） | `datagen` 連封存集數值都不生成，只生成 `ord_*` | `checks.check_archive_absent` |
| 2 禁止逐人標準化 | `features.fit_transform` 全世代 median/sd，`transform_one` 共用同一組 | `checks.check_no_per_patient_scaling`（單人 vs 批次逐值比對） |
| 3 Stage 1–3 只用 L1 集 | `pipeline.stage_feature_sets` | `checks.check_stage_features_L1` |
| 4 M0 先建先定版 | `models.M0Ledger`：`seal()` 前呼叫主模型效能即 raise | `checks.check_m0_sealed_before` |
| 5 未開立 ≠ 陰性 | 缺值一律 NaN，補值只在 CV 折內 | `checks.check_missing_is_nan` |
| 6 拒答是必要功能 | `models.pick_threshold`（訓練集鎖定）＋`apply_abstention` | `checks.check_report_completeness` |
| 7 無依據參數留 null | `params_io.check` 啟動警告，三處標示 | `pipeline` 起手即 assert `not chk["bad"]` |

## 檔案

- `pipeline.py`：主管線（資料 → Stage 0 → 特徵 → 分割 → **M0 定版** → M1/M2/M3 → 判定 → 鎖門檻 → 測試一次 → Stage 4 → Stage 5 → assert → 交付物）
- `src/`：`params_io`（出處檢核）、`seeding`、`datagen`（合成世代）、`features`（L/S/R 三讀法）、`models`（階梯／拒答／指標／M0 帳本）、`boxes`（分流格與建議）、`checks`（十一節八項）
- `results/`：`report.md`／`report.json`／`m0_baseline.json`／`thresholds.json`／`feature_importance.csv`／`limitations.md`／`figs/`
- `ui/index.html`：單一離線視覺化——六分頁（總覽／模型階梯與洩漏基準／拒答與測試表現／分流格與建議／**個案分流（輸入數值即時跑模型）**／參數出處）；前端不硬寫任何係數或門檻

## 主要結果（合成世代 n=4000，測試集只評估一次）

- **M0 洩漏基準**（只用「哪些檢驗被開立」）：Stage 1 0.81／Stage 2 0.79／Stage 3 0.71——開立行為本身就攜帶大量標籤資訊
- 三軸 M2 皆顯著超越 M0（+0.095 ~ +0.126，95% CI 下界皆 > 0）→ 無軸被移除
- 測試集作答者平衡正確率：急性度 0.961／部位 0.938／表現型 0.855（拒答率 0.034／0.048／0.190）
- Stage 4 分流格：判定率 0.750、作答者平衡正確率 0.789
- Stage 5：檢驗節省率 0.841、關鍵遺漏率 0.262、建議命中率 0.823；**判為 R 者五項強制建議完整率 1.00**（assert 保證），
  真 R 之遺漏 0.295 全部來自 Stage 4 誤分流（R 格召回 0.705）——照實報告，改善方向是 R 格召回而非放寬建議規則
- 形態欄位可得 0.23 < 0.30 閘門 → Stage 3 走代理特徵，已標明特異度下降並輸出兩子群比較
