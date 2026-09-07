# assumptions.md — SLE 減藥翻轉預警

## 界線值總表狀態

| 名稱 | status | 值 | 出處 |
|---|---|---|---|
| mu_c | anchored | 0.3849001794597505 | 數學常數 2/(3√3)：摺疊分岔臨界值（v2：模型尺度，不對應任何臨床門檻） |
| x_stable | anchored | [-1.0, 1.0] | 數學常數：μ=0 時雙穩態平衡點 |
| flare_rate_24m_range | anchored | {"win_lupus_discontinuation": 0.273, "derosa_withdrawal": 0.306, "win_lupus_continuation": 0.125} | WIN-Lupus（Jourde-Chiche 2022, PMID 35725295）12/44 vs 5/40；De Rosa 2018（PMID 30045812）11/36。v2 肆之六：用範圍不用單點 |
| flare_rate_52w | anchored | 0.32142857142857145 | Gopal 2023, PMID 37987842：9/28 於 52 週（含腎外發作，SFI） |
| biopsy_rule_counts | anchored | {"flared": 11, "flared_with_residual_activity": 10, "sensitivity_reported": 1.0, "specificity_reported": 0.88, "misclassification_reported": 0.083, "n_completed": 36} | De Rosa 2018, PMID 30045812：預測式含活性指數＋病程；v2 肆之四：只用於界定可信範圍，不當無誤差固定真值 |
| nih_ai_flare_threshold | anchored | 2 | De Rosa 2018, PMID 30045812：活性指數 > 2 者全部發作 |
| traj_egfr_proportions | anchored | {"stable": 0.879, "late_decline": 0.07, "persistent_decline": 0.051} | Ning 2026, PMID 41611465；v2 壹之二：僅作異質性背景證據，不直接移植為減藥模型參數 |
| flare_rate_by_traj_ning | anchored | {"late_decline": 0.91, "stable": 0.25} | Ning 2026, PMID 41611465（背景證據） |
| traj_slopes | pending_extraction | null（佔位 {"continuous_deterioration_x_per_year": 0.5, "sd": 0.15, "note": "佔位：連續惡化機制的平均水準每年上升 0.5（x 單位）；不得對外"}） | 待抽取：Ning 2026 全文 Fig1／補充表（PMC12863332；正文無數字表） |
| taper_schedule | pending_extraction | null（佔位 {"duration_days": 180, "shape": "linear", "complete": true, "presets": [{"name": "placeholder_fast", "duration_days": 90}, {"name": "placeholder_default", "duration_days": 180}, {"name": "placeholder_slow", "duration_days": 365}], "note": "佔位：自 t_onset 起線性遞減至 0"}） | 待抽取：EULAR 2025（PMID 41107121）全文『類固醇遞減與停藥、療程長度』；原始試驗（WIN-Lupus：2–3 年後停藥）與機構流程 |
| monitoring_items_and_interval | pending_extraction | null（佔位 {"default_interval_days": 30}） | 待抽取：EULAR 2025 全文『治療目標與里程碑』 |
| flare_definition_threshold | pending_extraction | null（佔位 {"x_threshold": 0.0, "note": "佔位：x 跨 0 記為『翻轉／跳轉』里程碑（t_jump）"}） | 待抽取：Gopal 2023 用 SELENA-SLEDAI 發作指數；原始工具 Petri 2005 NEJM（PMID 16354891） |
| mechanisms | study_defined | {"bifurcation": 0.22, "stochastic_escape": 0.04, "continuous_deterioration": 0.05, "exogenous_shock": 0.04, "noise_amplification": 0.05, "stable": 0.6} | 本研究設定（v2 肆之二 六種競爭機制）：分岔翻轉份額使總發作率落在 0.27–0.31（De Rosa 中 10/11 發作者有殘餘活性 → 多數發作為機制性）；連續惡化取 Ning 持續下降 5.1%；其餘為設定並掃描 |
| tau_x | study_defined | 30 | 本研究設定：系統鬆弛時間（v2：不沿用 CKD 版；掃描後由裁決定預設） |
| tau_xi | study_defined | 10 | 本研究設定：雜訊相關時間，須與 tau_x 分開識別（v2 肆之一） |
| sigma_xi | study_defined | 0.1 | 本研究設定：疾病過程波動強度，定義為緩解態 x 波動的定態 SD（x 單位）；OU 力振幅由 Var(x)=s²·τξ/(a(1+aτξ)) 反推 |
| meas_error_levels | study_defined | {"none": 0.0, "small": 0.05, "medium": 0.15, "large": 0.3} | 本研究設定：量測誤差 SD（x 單位）四階 |
| sampling_intervals | study_defined | {"daily": 1, "weekly": 7, "monthly": 30, "half_year": 182} | 本研究設定：取樣間隔四階 |
| irregular_sampling | study_defined | {"jitter_frac": 0.3, "missing_prob": 0.15} | 本研究設定：不規則回診（間隔抖動）與隨機缺失（v2 肆之三） |
| symptom_driven_sampling | study_defined | {"gain": 2.0} | 本研究設定：x 上升時回診機率上升（v2 肆之三 非隨機取樣） |
| treatment_intervention | study_defined | {"y_threshold": -0.4, "g_boost": 0.8, "delay_days": 14} | 本研究設定：觀測值異常後加藥（治療悖論情境，v2 肆之三／陸之五） |
| marker_map | study_defined | {"upcr": {"log_intercept": -1.6, "log_slope": 1.5, "detection_limit": 0.05}, "egfr": {"intercept": 95, "slope": -20}} | 本研究設定：潛在活性→臨床指標映射 h(k)（UPCR 對數線性、檢測下限；eGFR 線性）；僅供多指標情境 |
| adherence | study_defined | {"mean": 0.9, "sd": 0.1, "lapse_prob_per_month": 0.05, "lapse_days": 14} | 本研究設定：實際暴露＝處方×依從（v2 肆之五） |
| pkpd_lag_days | study_defined | 21 | 本研究設定：藥效延遲，g_eff 對處方暴露作一階滯後（v2 肆之五） |
| g0 | study_defined | 1.5 | 本研究設定：全劑量穩定效果；須使 μ_i − g0 < μ_c 對全體成立 |
| mu_bifurcation_margin | study_defined | [0.05, 0.6] | 本研究設定：分岔翻轉機制 μ_i − g_end 高於 μ_c 的均勻範圍（決定跨臨界時點分布） |
| mu_escape_margin | study_defined | [0.02, 0.15] | 本研究設定：隨機越障機制 μ_i − g_end 低於 μ_c 的均勻範圍（不達分岔點但能障小） |
| mu_low_dist | study_defined | {"mean": -0.9, "sd": 0.3} | 本研究設定：穩定／連續惡化／外生衝擊／雜訊放大機制的 μ_i 分布（遠低於 μ_c） |
| shock | study_defined | {"onset_range_after_taper": [30, 600], "mu_pulse": 1.5, "duration_days": 30} | 本研究設定：外生衝擊（感染等）以 μ 脈衝表示（v2 肆之二） |
| noise_amplification | study_defined | {"factor_end": 3.0} | 本研究設定：雜訊放大機制的 σξ 線性放大倍數（穩定性不變） |
| hazard | study_defined | {"lambda0_per_day": 0.00024447213772400535, "beta_per_x": 3.0} | 本研究設定：發作風險 λ(t)=λ0·exp(β·x)（β 使活動側風險遠高於緩解側）；λ0 於校準集校準使 24 個月總發作率落在 flare_rate_24m_range；λ0 於校準集（seed 20261818，n=800）校準達 24 個月 0.289；λ0 於校準集（seed 20261818，n=800）校準達 24 個月 0.289；λ0 於校準集（seed 20261818，n=3000）校準達 24 個月 0.289 |
| kappa | study_defined | 0.5 | 本研究設定：基線臨床特徵與 μ_i 的相關強度 |
| T | study_defined | 730 | 本研究設定：自 t_onset 起追蹤 24 個月（De Rosa／WIN-Lupus 主要終點） |
| run_in_days | study_defined | 90 | 本研究設定：減藥前觀察期 |
| N | study_defined | 3000 | 本研究設定：每個 Monte Carlo 資料集人數 |
| mc_datasets | study_defined | 5 | 本研究設定：獨立 Monte Carlo 資料集數（回報 MC 標準誤，v2 伍之六） |
| seed_families | study_defined | {"calibration": 1000, "test": 5000} | 本研究設定：校準集與測試集用不同 seed 族（v2 肆之六：校準與評估分離） |
| window_days | study_defined | 60 | 本研究設定：滾動窗長（時間單位；不規則取樣以時間窗計）。每週取樣下 30 天窗僅 4–5 點不足 min_obs=6（試跑確認指標全為 NaN），故預設 60 天並掃描 42–120；更短窗留給每日取樣情境 |
| min_obs_per_window | study_defined | 6 | 本研究設定：窗內最低觀測數（觀測密度關卡） |
| detrend | study_defined | "linear" | 本研究設定：事前指定去趨勢法（v2 伍之一） |
| gaussian_bw_frac | study_defined | 0.25 | 本研究設定 |
| joint_alarm | study_defined | "min_tau" | 本研究設定：病人層級聯合警報分數 S(t)（min_tau = AR(1) 與 SD 兩者 Kendall τ 之最小值；以全期最大值定閾，v2 伍之三） |
| false_alarm_budget_per_py | study_defined | 0.1 | 本研究設定（待與臨床者共同指定）：每病人年允許的假警報數；本設計每人只計首次警報，0.5/病人年在 2 年追蹤下等於允許約 87% 的穩定者觸發（試跑確認，形同無閾值），故預設 0.1（≈2 年內 18% 穩定者）並掃描 0.05–0.25；閾值於校準集鎖定 |
| surrogates | study_defined | "empirical_null" | 本研究設定：主要虛無 = 校準集穩定機制之真實序列（保留邊際、短期自相關、取樣與缺失、個體異質）；區塊置換等為敏感度（v2 伍之三、拾） |
| block_len_days | study_defined | 30 | 本研究設定：區塊置換區塊長（敏感度） |
| eval_every_days | study_defined | 7 | 本研究設定：聯合分數評估頻率 |
| landmarks | study_defined | [90, 180, 365, 540] | 本研究設定：自 t_onset 起算 |
| static_c_target | study_defined | [0.65, 0.75] | 本研究設定：靜態基線 C 的可信區間 |
| high_risk_top_share | study_defined | 0.2 | 本研究設定 |
| decision_thresholds | study_defined | [0.1, 0.2, 0.3, 0.4] | 本研究設定：決策曲線的風險門檻 |
| zero_signal_tolerance | study_defined | {"auc_band": [0.45, 0.55], "sens_over_fa_ratio_max": 1.5} | 本研究設定：零信號關卡容許範圍 |
| gate_max_delta_c_beta0 | study_defined | 0.01 | 本研究設定：結局脫耦時的最大 |ΔC| |
| k_max | study_defined | 8 | 本研究設定（分群為次要探索，v2 肆之二） |
| bic_improve_tol | study_defined | 0.05 | 本研究設定 |
| stability_min | study_defined | 0.6 | 本研究設定（Hennig） |
| bootstrap_n | study_defined | 200 | 本研究設定 |

## 世代生成設定（cohort.json）

| 名稱 | status | 設定 | 出處 |
|---|---|---|---|
| upcr_g_per_g | study_defined | {"dist": "lognormal", "median": 0.25, "sigma_log": 0.5, "clip": [0.02, 0.5]} | 本研究設定：完全緩解 <0.5 g/day 內的分布 |
| egfr | study_defined | {"dist": "normal", "mean": 95, "sd": 15, "clip": [45, 130]} | 本研究設定 |
| c3_mg_dl | study_defined | {"dist": "normal", "mean": 100, "sd": 20, "clip": [50, 160]} | 本研究設定 |
| anti_dsdna_iu_ml | study_defined | {"dist": "lognormal", "median": 30, "sigma_log": 1.0, "clip": [1, 800]} | 本研究設定 |
| disease_duration_yr | study_defined | {"dist": "lognormal", "median": 6.0, "sigma_log": 0.4, "clip": [3, 30]} | 本研究設定：下界由 WIN-Lupus（維持 2–3 年）與 De Rosa（≥36 個月＋≥12 個月緩解）納入條件推得 |
| prior_flares | study_defined | {"dist": "poisson", "lam": 1.0, "clip": [0, 6]} | 本研究設定 |
| chronicity_index | study_defined | {"dist": "poisson", "lam": 1.5, "clip": [0, 12]} | 本研究設定：Ning 2026 Table 1 切片時 CI 中位 0（0–2）供參；此為減藥前重複切片之慢性化 |

## ※ 佔位參數（4 項）：traj_slopes, taper_schedule, monitoring_items_and_interval, flare_definition_threshold

本次結果使用佔位參數，不得對外引用。

## 校準三步達成值

| 步驟 | 目標 | 達成 | 備註 |
|---|---|---|---|
| 一 λ0 | 24 個月發作率 ∈ [0.273, 0.306] | 0.289（λ0=2.445e-04） | 52 週 0.165 vs 0.32142857142857145；機制別 {'bifurcation': 0.9811066126855601, 'continuous_deterioration': 0.0, 'exogenous_shock': 0.044642857142857144, 'noise_amplification': 0.1625, 'stable': 0.005777007510109763, 'stochastic_escape': 0.8} |
| 二 切片臂 | sens/spec ∈ 95% CI {'sensitivity': (0.7150858470818455, 1.0), 'specificity': (0.6878096927137647, 0.9745346033522668), 'counts': {'tp': 11, 'fn': 0, 'tn': 22, 'fp': 3}} | εB=0.4, thr=0.189 → 0.942/0.860 |  |
| 三 靜態 C | 0.65–0.75 | 0.711（κ=0.5） | None |

## 品質關卡（v2 六道）

| 關卡 | 值 | 條件 | 結果 |
|---|---|---|---|
| Q1 機制獨立 | {"signatures": {"score_cohort": true, "noninvasive_features": true, "noninvasive_arm": true}, "source_clean": true, "hash_invariant": true} | 簽章無 μ_i／mech／B；打亂後雜湊不變 | 通過 |
| Q2 零信號 | {"alarm_rate": 0.21216666666666667, "alarm_rate_event": 0.2277379733879222, "alarm_rate_nonevent": 0.209137965359347, "ratio": 1.0889365448144057, "dynamic_auc": [0.49550023490989636, 0.503481866266269, 0.5133378905792791, 0.4976205813003226, 0.5012743551255577, 0.48846794639147056, 0.50229437424389 | 警報率比 ≤ 1.5；AUC ∈ [0.45, 0.55]；max_L|mean ΔC| < 0.01（3 seeds 平均） | 通過 |
| Q3 閾值鎖定 | {"calibration_seeds": [20261818], "test_seeds": [20265818, 20265819, 20265820, 20265821, 20265822], "disjoint": true, "locked_hash": "fb43e1285a6a167f"} | 校準與測試 seed 族不交集；閾值檔雜湊於測試前寫定 | 通過 |
| Q4 時間方向 | {"cut_day": 455, "identical_before_cut": true} | 打亂未來後，切點前分數雜湊相同 | 通過 |
| Q5 觀測密度 | {"coverage_by_eval": [0.0, 0.0, 0.999, 0.999, 0.999, 0.999, 0.999, 0.998, 0.997, 0.997, 0.996, 0.995, 0.994, 0.993, 0.992, 0.989, 0.987, 0.985, 0.982, 0.978, 0.972, 0.968, 0.961, 0.951, 0.939, 0.932, 0.921, 0.909, 0.898, 0.885, 0.872, 0.861, 0.844, 0.83, 0.821, 0.813, 0.802, 0.794, 0.789, 0.78, 0.77 | 指標不可算處為 NaN 且不觸發警報（由 first_alarm 之 NaN 語意保證）；回報覆蓋率 | 通過 |
| Q6 結局盲化 | "not_applicable（合成資料階段；Stage 2/3 真實資料時：計算指標者不得看結局）" | — | 通過 |
