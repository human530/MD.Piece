# SLE 狼瘡腎炎減藥翻轉預警（計畫書 v2 版模型）

依 `docs/研究計畫書_v2.md`（2026-08-18，主）與 `docs/研究計畫書_v1.md`（輔）建置；工作規則沿用
`docs/ClaudeCode_全案建置指示_v1.md`（thresholds.json 唯一數值真相、seeding、每個數字有出處、不可達留白）。

```bash
cd experiments/sle_taper
python -m pytest tests -q            # 12 項單元測試
python run.py --phase calibrate      # 校準集：λ0／切片臂 (εB,閾值)／κ → results/locked.json、assumptions.md
python run.py --phase default        # 測試集：預設情境 × MC 資料集＋六道品質關卡＋觀測情境 → results/default.json
python run.py --phase grid           # 敏感度格點（一次一軸）＋可辨識邊界 → results/grid.json
python run.py --verify [--quick]     # 驗收清單
```

## 計畫書 v2 → 程式對照

| 計畫書 v2 | 程式 |
|---|---|
| 肆之一 模型（τx／τξ／σξ 分開；μ(t)=μᵢ−g_eff(t)） | `m1_generator.simulate_cohort`、`ou_unit`（單位 OU × 振幅，σξ=緩解態 x 定態 SD） |
| 肆之二 六種競爭機制（分岔／隨機越障／連續惡化／外生衝擊／雜訊放大／穩定） | `m1_generator.MECHS`；機制標籤先於動力學；比例 `thresholds.mechanisms`（含掃描） |
| 肆之三 觀測模型（規則／不規則／症狀驅動取樣、量測誤差、檢測下限、治療悖論） | `m1_generator.observe`、`treatment_intervention`；情境 `cohort.observation_scenarios` |
| 肆之四 切片評分 B=q[x(t₀),μᵢ,CI,duration]+εB；效能=可信範圍非固定真值 | `m6_arms.biopsy_plausible_range`（De Rosa 計數→Clopper–Pearson 95% CI）、`calibrate_biopsy` |
| 肆之五 減藥=處方×依從×藥動延遲 | `prescribed_schedule`×`adherence_paths`→`effective_exposure`（一階滯後） |
| 肆之六 校準（發作率用範圍 0.273–0.306；校準／測試 seed 族分離） | `calibrate.step1_lambda0`（λ0 二分）、`seed_families`；`run_cell` 校準集鎖閾→測試集只讀 |
| 伍之一～三 預警指標、聯合警報分數 S(t)、假警報負擔定閾（不用 p 值） | `m5_ews`：`rolling_indicators`＋`prefix_kendall`→`joint_score(min_tau)`→`lock_threshold`；替代虛無 `surrogate_threshold` |
| 伍之四 主要指標（固定負擔下各機制敏感度／提前期／假警報率；水準規則對照） | `m5_ews.evaluate`；`score_cohort` 同時輸出 level／trend 規則 |
| 伍之五 次要指標（AUROC、Brier、校準斜率截距、決策曲線、NNT、探索性再分類） | `m4_predict.secondary_metrics`、`run_predict` |
| 伍之六 MC 標準誤 | `run.summarize_cell`（mc_datasets=5） |
| 陸 六道品質關卡（機制獨立／零信號／閾值鎖定／時間方向／觀測密度／結局盲化） | `gates.q1`–`q6`（Q6 於合成階段標 not_applicable） |
| 拾 敏感度清單＋可辨識邊界 | `run.phase_grid`（axes＋identifiability：τx×取樣間隔×量測誤差） |

v1 之 H4（精度×頻率交換率）由 grid 的 meas_error×interval 軸與兩臂比較（`m6_arms.compare_arms`，不合成單一分數）覆蓋。
v1 四道閘門中的 G1／G3 不在 v2 內（v2 以機制標籤取代分群、以治療悖論情境正面處理減藥前訊號），G2→Q2、G4→Q1。

- `params/thresholds.json`：唯一數值真相（value/source/status；pending_extraction 佔位三處標示；`_meta.design_changes_after_smoke_test` 記錄試跑後、正式跑前的規格退化修正）
- 亂數一律經 `seeding.module_rng(master_seed, module, sub)`；禁 `np.random` 全域介面
- `ui/`：`python ui/export_ui_pack.py` 產生兩個單一離線 HTML——`index.html`（視覺化四層＋「模擬」分頁頁內跑同構模型）與 `assess.html`（評估工作台：填基線數值＋監測序列 → 靜態發作機率＋動態警報條件，係數/門檻全由校準流程匯出）；前端不硬寫任何係數
- 佔位參數（補齊前產出不得對外）：Ning 斜率、EULAR 減藥排程與監測、SELENA-SLEDAI 發作操作型門檻
